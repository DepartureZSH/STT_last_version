import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict
from itertools import combinations, product
import time
import torch
from tqdm import tqdm
from src.utils.dataReader import PSTTReader
from src.utils.validator import solution
from src.utils.solutionWriter import export_solution_xml

class MIPSolver:
    def __init__(self, reader, logger, config=None):
        """初始化求解器"""
        self.reader = reader
        self.logger = logger
        self.time_limit = config['train']['MIP']['time_limit']
        self.Threads = config['train']['MIP']['Threads']
        self.MIPGap = config['train']['MIP']['MIPGap']
        self.PoolSolutions = config['train']['MIP']['PoolSolutions']
        
        # Gurobi模型
        self.model = gp.Model("MIPSolver")
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('Threads', self.Threads)
        self.model.setParam('MIPGap', self.MIPGap)  # 1% gap
        self.model.setParam('MIPFocus', 1)  # 专注于找到可行解
        self.model.setParam('PoolSolutions', self.PoolSolutions)
        
        # 决策变量
        self.x = {}  # x[cid, time_idx, room_id]: 课程-时间-教室
        self.y = {}  # 
        self.w = {}  # 
        self.u = {} # x[cid]: 未分配课程
        
        # 辅助变量
        self.penalty_vars = []  # [(var, cost), ...]
        
        # 冲突图（简化版本，不做完整的图预处理）
        self.time_conflicts = defaultdict(set)  # {(c1, t1): {(c2, t2), ...}}
        
        # 索引映射
        self.class_to_time_options = {}  # {cid: [(time_option, time_idx), ...]}
        self.class_to_room_options = {}  # {cid: [room_id, ...]}
        self.class_to_valid_options = {} # [cid, time_idx, room_id]: True

        # 缓存：预计算时间冲突
        self.time_conflict_cache = {}  # {(bits1, bits2): bool}

        # Hybrid solver: fixed-assignment tracking
        self._fix_constraints: list = []         # Gurobi constraint objects added by fix_assignments()
        self._fix_constr_to_cid: dict = {}       # {constr_name: cid}
        self._fixed_assignments: dict = {}       # {cid: (tidx, rid)} currently pinned

        # Divide-and-conquer submodel support
        # Set by build_submodel(); None means "use all classes / all constraints"
        self._active_classes: set = None         # subset of class IDs to model
        self._active_hard_constraints: list = None  # hard constraints to enforce
        self._active_soft_constraints: list = None  # soft constraints to enforce
        self._forbid_constraints: list = []      # constraints added by forbid_time_room()

        self.logger.info(f"Initialized solver for problem: {self.reader.problem_name}")
        self.logger.info(f"Classes: {len(self.reader.classes)}, Rooms: {len(self.reader.rooms)}")
        self.logger.info(f"Hard constraints: {len(self.reader.distributions['hard_constraints'])}")
        self.logger.info(f"Soft constraints: {len(self.reader.distributions['soft_constraints'])}")
        
    def build_model(self):
        """构建完整的MIP模型"""
        print("\n=== Building MIP Model ===")
        
        # 1. 预处理：建立索引
        self._build_indices()
        # 预构建教室冲突图
        # self._build_room_conflict_graph()

        # 2. 创建变量
        self._create_variables()
        
        # 3. 添加基础约束
        self._add_primary_constraints()
        
        # 4. 添加分布约束
        self._add_distribution_constraints()
        
        # 5. 设置目标函数
        self._set_objective()
        
        # self.logger.info(f"Model built: {self.model.NumVars} vars, {self.model.NumConstrs} constrs")
        
    def _build_indices(self):
        """构建课程到时间/教室选项的索引"""
        print("Building indices...")
        
        active = self._active_classes  # None → all classes
        for cid, class_data in self.reader.classes.items():
            if active is not None and cid not in active:
                continue
            # 时间选项
            time_options = []
            for idx, topt in enumerate(class_data['time_options']):
                time_options.append((topt, idx))
            self.class_to_time_options[cid] = time_options
            
            # 教室选项
            room_options = []
            for ropt in class_data['room_options']:
                room_options.append(ropt['id'])
            
            # 如果不需要教室，添加虚拟教室
            if not class_data['room_required']:
                room_options = ['dummy']
            self.class_to_room_options[cid] = room_options

            options = []
            for topt, tidx in time_options:
                for rid in room_options:
                    self.class_to_valid_options[cid, tidx, rid] = True
            
        print(f"Indexed {len(self.class_to_time_options)} classes")
    
    def _build_room_conflict_graph(self):
        """
        预构建教室冲突图（可选的进一步优化）
        在 _build_indices 后调用
        """
        print("Building room conflict graph...")
        
        self.room_conflicts = defaultdict(set)  # {rid: {(c1, t1, c2, t2), ...}}
        
        for rid in tqdm(self.reader.rooms.keys()):
            classes_using_room = [
                cid for cid, rooms in self.class_to_room_options.items()
                if rid in rooms
            ]
            
            if len(classes_using_room) < 2:
                continue
            
            for i, c1 in enumerate(classes_using_room):
                for c2 in classes_using_room[i+1:]:
                    time_opts1 = self.class_to_time_options[c1]
                    time_opts2 = self.class_to_time_options[c2]
                    for topt1, tidx1 in time_opts1:
                        bits1 = topt1['optional_time_bits']
                        # 检查课程1的这个时间在这个教室是否可用
                        if not self._is_room_available(rid, time_bits=bits1):
                            continue

                        for topt2, tidx2 in time_opts2:
                            bits2 = topt2['optional_time_bits']

                            # 检查课程2的这个时间在这个教室是否可用
                            if not self._is_room_available(rid, time_bits=bits2):
                                continue
                            
                            if self._times_conflict(bits1, bits2):
                                self.room_conflicts[rid].add((c1, tidx1, c2, tidx2))
        
        total_conflicts = sum(len(v) for v in self.room_conflicts.values())
        print(f"Found {total_conflicts} room conflicts across {len(self.room_conflicts)} rooms")

    def _create_variables(self):
        """创建决策变量"""
        print("Creating variables...")

        var_count = 0
        filtered_count = 0

        active = self._active_classes
        for cid in self.reader.classes.keys():
            if active is not None and cid not in active:
                continue
            time_options = self.class_to_time_options[cid]
            room_options = self.class_to_room_options[cid]
            self.u[cid] = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"u_{cid}"
            )

            # 创建 x[cid, time_idx, room_id]
            # for i, (tidx, topt, rid, valid) in enumerate(options):
            for topt, tidx in time_options:
                time_bits = topt['optional_time_bits']
                for rid in room_options:
                    # 检查这个时间-教室组合是否可用
                    if rid != 'dummy' and not self._is_room_available(rid, time_bits=time_bits):
                        # 跳过不可用的组合
                        self.class_to_valid_options[cid, tidx, rid] = False
                        filtered_count += 1
                        continue

                    var_name = f"x_{cid}_{tidx}_{rid}"
                    self.x[cid, tidx, rid] = self.model.addVar(
                        vtype=GRB.BINARY, 
                        name=var_name
                    )
                    var_count += 1
            
                # 创建 y[cid, time_idx]
                var_name = f"y_{cid}_{tidx}"
                self.y[cid, tidx] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=var_name
                )
            
            # 创建 w[cid, room_id]
            for rid in room_options:
                var_name = f"w_{cid}_{rid}"
                self.w[cid, rid] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=var_name
                )
        
        self.logger.info(f"Created {var_count} x variables")
        self.logger.info(f"Filtered out {filtered_count} unavailable time-room combinations")
        self.logger.info(f"Created {len(self.y)} y variables, {len(self.w)} w variables")
        self.logger.info(f"Created {var_count} x variables, {len(self.y)} y variables, {len(self.w)} w variables")

    def _is_room_available(self, room_id, time_mat=None, time_bits=None):
        """
        检查教室在给定时间是否可用
        
        Args:
            room_id: 教室ID
            time_bits: (weeks_bits, days_bits, start, length)
        
        Returns:
            bool: True if 教室可用
        """
        if room_id not in self.reader.rooms:
            return True
        
        room_data = self.reader.rooms[room_id]
        unavailables = room_data.get('unavailables_bits', [])
        unavailable_zip = room_data.get('unavailable_zip', None)
        
        if not unavailables:
            return True
        
        if time_bits == None:
            if self._time_matrix_overlap(unavailable_zip, time_mat):
                return False
            return True
        
        else:
            # 检查是否与任何不可用时间冲突
            for unavail_bits in unavailables:
                unavail_weeks, unavail_days, unavail_start, unavail_length = unavail_bits
                
                if unavail_weeks is None or unavail_days is None:
                    continue
                if unavail_start is None or unavail_length is None:
                    continue
                
                if self._time_conflicts_with_unavailable(time_bits, unavail_bits):
                    return False
            
            return True

    def _add_primary_constraints(self):
        """添加基础调度约束"""
        print("Adding primary constraints...")

        active = self._active_classes

        # 1. 每个课程必须分配恰好一个时间和教室
        for cid in self.reader.classes.keys():
            if active is not None and cid not in active:
                continue
            time_options = self.class_to_time_options[cid]
            room_options = self.class_to_room_options[cid]

            valid_x_vars = []
            for _, tidx in time_options:
                for rid in room_options:
                    if (cid, tidx, rid) in self.x:
                        valid_x_vars.append(self.x[cid, tidx, rid])

            if valid_x_vars:
                self.model.addConstr(
                    gp.quicksum(valid_x_vars) + self.u[cid] == 1,
                    name=f"assign_{cid}"
                )
            else:
                self.model.addConstr(
                    self.u[cid] == 1,
                    name=f"assign_unavail_{cid}"
                )
                self.logger.info(f"Warning: Class {cid} has no valid time-room combinations!")

        # 2. 链接 x 和 y 变量
        for cid in self.reader.classes.keys():
            if active is not None and cid not in active:
                continue
            time_options = self.class_to_time_options[cid]
            room_options = self.class_to_room_options[cid]

            for _, tidx in time_options:
                x_vars = [self.x[cid, tidx, rid]
                          for rid in room_options
                          if (cid, tidx, rid) in self.x]
                if x_vars:
                    self.model.addConstr(
                        gp.quicksum(x_vars) == self.y[cid, tidx],
                        name=f"link_y_{cid}_{tidx}"
                    )

        # 3. 链接 x 和 w 变量
        for cid in self.reader.classes.keys():
            if active is not None and cid not in active:
                continue
            time_options = self.class_to_time_options[cid]
            room_options = self.class_to_room_options[cid]

            for rid in room_options:
                x_vars = [self.x[cid, tidx, rid]
                          for _, tidx in time_options
                          if (cid, tidx, rid) in self.x]
                if x_vars:
                    self.model.addConstr(
                        gp.quicksum(x_vars) == self.w[cid, rid],
                        name=f"link_w_{cid}_{rid}"
                    )
                # self.model.addConstr(
                #     gp.quicksum(self.x[cid, tidx, rid] for _, tidx in time_options if (cid, tidx, rid) in self.x)
                #     == self.w[cid, rid],
                #     name=f"link_w_{cid}_{rid}"
                # )
        # # 4. 教室不可用时间约束（新增）
        # self._add_room_unavailable_constraints()

        # 5. 教室不能双重预订（使用预构建的冲突图）
        if hasattr(self, 'room_conflicts'):
            self._add_room_capacity_constraints_from_graph()
        else:
            self._add_room_capacity_constraints()
        
        print("Primary constraints added")
    
    def _add_room_unavailable_constraints(self):
        """
        添加教室不可用时间约束
        如果教室在某个时间段不可用，则不能在该时间段分配课程到该教室
        """
        print("Adding room unavailable constraints...")
        
        constraint_count = 0
        
        for rid, room_data in self.reader.rooms.items():
            unavailables = room_data.get('unavailables_bits', [])
            
            if not unavailables:
                continue
            
            # 找到所有可能使用该教室的课程
            classes_using_room = [
                cid for cid, rooms in self.class_to_room_options.items()
                if rid in rooms
            ]
            
            if not classes_using_room:
                continue
            
            # 对每个不可用时间段
            for unavail_weeks, unavail_days, unavail_start, unavail_length in unavailables:
                if unavail_weeks is None or unavail_days is None:
                    continue
                if unavail_start is None or unavail_length is None:
                    continue
                
                unavail_end = unavail_start + unavail_length
                
                # 检查每个可能使用该教室的课程
                for cid in classes_using_room:
                    time_opts = self.class_to_time_options[cid]
                    
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        
                        # 检查课程时间是否与不可用时间冲突
                        if self._time_conflicts_with_unavailable(
                            bits, 
                            (unavail_weeks, unavail_days, unavail_start, unavail_length)
                        ):
                            # 禁止在这个时间使用这个教室
                            if (cid, tidx, rid) in self.x:
                                self.model.addConstr(
                                    self.x[cid, tidx, rid] == 0,
                                    name=f"unavail_{rid}_{cid}_{tidx}"
                                )
                                constraint_count += 1
        
        print(f"Added {constraint_count} room unavailable constraints")

    def _time_conflicts_with_unavailable(self, time_bits, unavail_bits):
        """
        检查课程时间是否与教室不可用时间冲突
        
        Args:
            time_bits: (weeks_bits, days_bits, start, length) - 课程时间
            unavail_bits: (weeks_bits, days_bits, start, length) - 不可用时间
        
        Returns:
            bool: True if 有冲突
        """
        class_weeks, class_days, class_start, class_length = time_bits
        unavail_weeks, unavail_days, unavail_start, unavail_length = unavail_bits
        
        # 检查weeks是否有交集
        weeks_int1 = int(class_weeks, 2)
        weeks_int2 = int(unavail_weeks, 2)
        if (weeks_int1 & weeks_int2) == 0:
            return False
        
        # 检查days是否有交集
        days_int1 = int(class_days, 2)
        days_int2 = int(unavail_days, 2)
        if (days_int1 & days_int2) == 0:
            return False
        
        # 检查时间段是否重叠
        class_end = class_start + class_length
        unavail_end = unavail_start + unavail_length
        
        if not ((class_start < unavail_end) and (unavail_start < class_end)):
            return False
        
        return True

    def _add_room_capacity_constraints(self):
        """添加教室容量约束（防止双重预订）"""
        print("Adding room capacity...")
        constraint_count = 0
        
        # 对每个真实教室
        for rid in tqdm(self.reader.rooms.keys()):
            # 找到所有可能使用该教室的课程
            classes_using_room = []
            for cid, rooms in self.class_to_room_options.items():
                if rid in rooms:
                    classes_using_room.append(cid)
            
            if len(classes_using_room) < 2:
                continue
            
            # 对每对可能使用同一教室的课程，检查时间冲突
            for i, c1 in enumerate(classes_using_room):
                for c2 in classes_using_room[i+1:]:
                    # 获取两个课程的所有时间选项
                    time_opts1 = self.class_to_time_options[c1]
                    time_opts2 = self.class_to_time_options[c2]
                    
                    # 检查所有时间选项对是否有冲突
                    for topt1, tidx1 in time_opts1:
                    # for tidx1, topt1, tidx1, valid1 in options1:
                        if not self.class_to_valid_options[c1, tidx1, rid]:
                            continue
                        bits1 = topt1['optional_time_bits']
                        
                        for topt2, tidx2 in time_opts2:
                            if not self.class_to_valid_options[c2, tidx2, rid]:
                                continue
                            bits2 = topt2['optional_time_bits']
                            
                            # 使用位运算快速检查时间冲突
                            if self._times_conflict(bits1, bits2):
                                # 如果两个课程的这两个时间选项冲突，
                                # 则不能同时在这个教室使用
                                if (c1, tidx1, rid) in self.x and (c2, tidx2, rid) in self.x:
                                    self.model.addConstr(
                                        self.x[c1, tidx1, rid] + self.x[c2, tidx2, rid] <= 1,
                                        name=f"room_conflict_{rid}_{c1}_{tidx1}_{c2}_{tidx2}"
                                    )
                                    constraint_count += 1
    
    def _add_room_capacity_constraints_from_graph(self):
        """
        从预构建的冲突图添加约束（最快的方法）
        """
        print("Adding room capacity constraints from conflict graph...")
        
        constraint_count = 0
        
        for rid, conflicts in self.room_conflicts.items():
            for c1, tidx1, c2, tidx2 in conflicts:
                if (c1, tidx1, rid) in self.x and (c2, tidx2, rid) in self.x:
                    self.model.addConstr(
                        self.x[c1, tidx1, rid] + self.x[c2, tidx2, rid] <= 1,
                        name=f"room_{rid}_c{c1}t{tidx1}_c{c2}t{tidx2}"
                    )
                    constraint_count += 1
        
        print(f"Added {constraint_count} room capacity constraints")

    def _time_covers_slot(self, time_bits, week, day, slot):
        """检查时间选项是否覆盖特定的时间槽"""
        weeks_bits, days_bits, start, length = time_bits
        
        # 使用位运算快速检查
        if week < len(weeks_bits) and weeks_bits[week] == '1':
            if day < len(days_bits) and days_bits[day] == '1':
                if start <= slot < start + length:
                    return True
        
        return True
    
    def _times_conflict(self, time_bits1, time_bits2):
        """
        使用位运算快速检查两个时间是否冲突，带缓存
        """
        # 创建缓存键（确保顺序一致）
        if time_bits1 < time_bits2:
            cache_key = (time_bits1, time_bits2)
        else:
            cache_key = (time_bits2, time_bits1)
        
        # 检查缓存
        if cache_key in self.time_conflict_cache:
            return self.time_conflict_cache[cache_key]
        
        week_bits1, day_bits1, start1, length1 = time_bits1
        week_bits2, day_bits2, start2, length2 = time_bits2
        
        end1 = start1 + length1
        end2 = start2 + length2
        
        # 检查时间段是否重叠
        if not ((start1 < end2) and (start2 < end1)):
            self.time_conflict_cache[cache_key] = False
            return False
        
        # 使用位运算检查days是否有交集
        days_int1 = int(day_bits1, 2)
        days_int2 = int(day_bits2, 2)
        and_days = days_int1 & days_int2
        
        if and_days == 0:
            self.time_conflict_cache[cache_key] = False
            return False
        
        # 使用位运算检查weeks是否有交集
        week_int1 = int(week_bits1, 2)
        week_int2 = int(week_bits2, 2)
        and_week = week_int1 & week_int2
        
        if and_week == 0:
            self.time_conflict_cache[cache_key] = False
            return False
        
        self.time_conflict_cache[cache_key] = True
        return True
    
    def _times_overlap(self, time_bits1, time_bits2):
        """为了兼容性，保留这个方法"""
        return self._times_conflict(time_bits1, time_bits2)
    
    def _time_matrix_overlap(self, time_matrix1, time_matrix2):
        overlap = torch.logical_and(time_matrix1, time_matrix2)
        return torch.any(overlap).item()

    def _check_travel_conflict(self, bits1, bits2, travel_times):
        """检查两个时间是否可能需要考虑旅行时间"""
        weeks1, days1, start1, length1 = bits1
        weeks2, days2, start2, length2 = bits2
        
        # 检查是否有共同的week和day
        weeks_int1 = int(weeks1, 2)
        weeks_int2 = int(weeks2, 2)
        if (weeks_int1 & weeks_int2) == 0:
            return False
        
        days_int1 = int(days1, 2)
        days_int2 = int(days2, 2)
        if (days_int1 & days_int2) == 0:
            return False
        
        # 如果在同一天，检查时间是否紧邻
        end1 = start1 + length1
        end2 = start2 + length2
        
        # 时间紧邻或有小间隙（需要考虑旅行）
        gap = min(abs(start2 - end1), abs(start1 - end2))
        
        return gap >= 0 and gap < 60  # 假设60个时间槽内需要考虑旅行

    def _get_travel_time(self, room1, room2, travel_times):
        """获取两个教室之间的旅行时间"""
        if not travel_times:
            return 0
        
        if str(room1) in travel_times and str(room2) in travel_times[str(room1)]:
            return travel_times[str(room1)][str(room2)]
        
        return 0

    def _has_enough_travel_time(self, bits1, bits2, travel_time):
        """检查两个时间之间是否有足够的旅行时间"""
        _, _, start1, length1 = bits1
        _, _, start2, length2 = bits2
        
        end1 = start1 + length1
        end2 = start2 + length2
        
        # 计算间隙
        if start2 >= end1:
            gap = start2 - end1
        elif start1 >= end2:
            gap = start1 - end2
        else:
            return False  # 重叠
        
        return gap >= travel_time

    def _add_distribution_constraints(self):
        """添加分布约束"""
        print("Adding distribution constraints...")

        # Use override lists when building a submodel; fall back to reader data
        hard_list = (
            self._active_hard_constraints
            if self._active_hard_constraints is not None
            else self.reader.distributions['hard_constraints']
        )
        soft_list = (
            self._active_soft_constraints
            if self._active_soft_constraints is not None
            else self.reader.distributions['soft_constraints']
        )

        for constraint in tqdm(hard_list, total=len(hard_list)):
            self._add_single_distribution_constraint(constraint, is_hard=True)

        for constraint in tqdm(soft_list, total=len(soft_list)):
            self._add_single_distribution_constraint(constraint, is_hard=False)

        print(f"Distribution constraints added, {len(self.penalty_vars)} penalty variables")
    
    def _add_single_distribution_constraint(self, constraint, is_hard):
        """添加单个分布约束"""
        ctype = constraint['type']
        # When running as a submodel, filter out classes not indexed
        classes = [c for c in constraint['classes'] if c in self.class_to_time_options]
        if not classes:
            return
        penalty = constraint.get('penalty', 0)
        
        # 根据约束类型调用相应的处理函数
        if ctype == 'SameTime':
            self._add_same_time_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentTime':
            self._add_different_time_constraint(classes, is_hard, penalty)
        elif ctype == 'SameRoom':
            self._add_same_room_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentRoom':
            self._add_different_room_constraint(classes, is_hard, penalty)
        elif ctype == 'NotOverlap':
            self._add_not_overlap_constraint(classes, is_hard, penalty)
        elif ctype == 'Overlap':
            self._add_overlap_constraint(classes, is_hard, penalty)
        elif ctype == 'SameStart':
            self._add_same_start_constraint(classes, is_hard, penalty)
        elif ctype == 'SameDays':
            self._add_same_days_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentDays':
            self._add_different_days_constraint(classes, is_hard, penalty)
        elif ctype == 'SameWeeks':
            self._add_same_weeks_constraint(classes, is_hard, penalty)
        elif ctype == 'DifferentWeeks':
            self._add_different_weeks_constraint(classes, is_hard, penalty)
        elif ctype == 'SameAttendees':
            self._add_same_attendees_constraint(classes, is_hard, penalty)
        elif ctype == 'Precedence':
            self._add_precedence_constraint(classes, is_hard, penalty)
        elif ctype.startswith('MinGap'):
            min_gap = int(ctype.split('(')[1].rstrip(')'))
            self._add_min_gap_constraint(classes, min_gap, is_hard, penalty)
        elif ctype.startswith('MaxDays'):
            max_days = int(ctype.split('(')[1].rstrip(')'))
            self._add_max_days_constraint(classes, max_days, is_hard, penalty)
        elif ctype.startswith('MaxDayLoad'):
            max_slots = int(ctype.split('(')[1].rstrip(')'))
            self._add_max_day_load_constraint(classes, max_slots, is_hard, penalty)
        elif ctype.startswith('WorkDay'):
            max_slots = int(ctype.split('(')[1].rstrip(')'))
            self._add_workday_constraint(classes, max_slots, is_hard, penalty)
        elif ctype.startswith('MaxBreaks'):
            # 解析 MaxBreaks(R,S) 格式
            params = ctype.split('(')[1].rstrip(')').split(',')
            max_breaks = int(params[0])
            min_break_length = int(params[1])
            self._add_max_breaks_constraint(classes, max_breaks, min_break_length, is_hard, penalty)
        elif ctype.startswith('MaxBlock'):
            # 解析 MaxBlock(M,S) 格式
            params = ctype.split('(')[1].rstrip(')').split(',')
            max_block_length = int(params[0])
            max_gap_in_block = int(params[1])
            self._add_max_block_constraint(classes, max_block_length, max_gap_in_block, is_hard, penalty)
        else:
            print(f"Warning: Constraint type '{ctype}' not implemented")
    
    def _add_same_start_constraint(self, classes, is_hard, penalty):
        """
        SameStart: 课程必须在相同的开始时间
        即：start time必须相同
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    if topt1['optional_time_bits'][2] != topt2['optional_time_bits'][2]:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_same_time_constraint(self, classes, is_hard, penalty):
        """SameTime: 所有课程必须在相同时间"""
        if len(classes) < 2:
            return
        
        # 对每对课程
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            # 找到时间完全相同的选项对
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    _, _, start1, end1 = topt1["optional_time_bits"]
                    _, _, start2, end2 = topt2["optional_time_bits"]
                    if start1 <= start2 and start2 + end2 <= start1 + end1:
                        continue
                    elif start2 <= start1 and start1 + end1 <= start2 + end2:
                        continue
                    else:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1,
                                name=f"same_time_{c1}_{c2}_{tidx1}_{tidx2}"
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))
    
    def _add_different_time_constraint(self, classes, is_hard, penalty):
        """DifferentTime: 课程必须在不同时间"""
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            # 找到时间相同的选项对，禁止同时选择
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    _, _, start1, end1 = topt1['optional_time_bits']
                    _, _, start2, end2 = topt2['optional_time_bits']
                    if (start1 + end1 <= start2) or (start2 + end2 <= start1):
                        continue
                    else:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))
    
    def _add_same_days_constraint(self, classes, is_hard, penalty):
        """
        SameDays: 课程必须在相同的星期几
        即：days_bits必须完全相同
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    day_bits1 = topt1['optional_time_bits'][1]
                    days_int1 = int(day_bits1, 2)
                    day_bits2 = topt2['optional_time_bits'][1]
                    days_int2 = int(day_bits2, 2)
                    or_ = days_int1 | days_int2
                    if not (or_ == days_int1 or or_ == days_int2):
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_different_days_constraint(self, classes, is_hard, penalty):
        """
        DifferentDays: 课程必须在不同的星期几
        即：days_bits不能有交集
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    day_bits1 = topt1['optional_time_bits'][1]
                    days_int1 = int(day_bits1, 2)
                    day_bits2 = topt2['optional_time_bits'][1]
                    days_int2 = int(day_bits2, 2)
                    and_ = days_int1 & days_int2
                    if not and_ == 0:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_same_weeks_constraint(self, classes, is_hard, penalty):
        """
        SameWeeks: 课程必须在相同的周
        即：weeks_bits必须完全相同
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1 = topt1['optional_time_bits'][0]
                    week_int1 = int(week_bits1, 2)
                    week_bits2 = topt2['optional_time_bits'][0]
                    week_int2 = int(week_bits2, 2)
                    or_ = week_int1 | week_int2
                    if not (or_ == week_int1 or or_ == week_int2):
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_different_weeks_constraint(self, classes, is_hard, penalty):
        """
        DifferentWeeks: 课程必须在不同的周
        即：weeks_bits不能有交集
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1 = topt1['optional_time_bits'][0]
                    week_int1 = int(week_bits1, 2)
                    week_bits2 = topt2['optional_time_bits'][0]
                    week_int2 = int(week_bits2, 2)
                    and_ = week_int1 & week_int2
                    if not and_ == 0:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))
    
    def _add_same_room_constraint(self, classes, is_hard, penalty):
        """SameRoom: 课程必须在同一教室"""
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            # 对每个教室，要么都选，要么都不选
            rooms1 = self.class_to_room_options[c1]
            rooms2 = self.class_to_room_options[c2]
            
            common_rooms = set(rooms1) & set(rooms2)
            
            if is_hard:
                # 必须选择共同教室
                for r in common_rooms:
                    self.model.addConstr(
                        self.w[c1, r] == self.w[c2, r],
                        name=f"same_room_{c1}_{c2}_{r}"
                    )
            else:
                # 软约束
                for r in common_rooms:
                    p = self.model.addVar(vtype=GRB.BINARY)
                    self.model.addConstr(self.w[c1, r] - self.w[c2, r] <= p)
                    self.model.addConstr(self.w[c2, r] - self.w[c1, r] <= p)
                    self.penalty_vars.append((p, penalty))
    
    def _add_different_room_constraint(self, classes, is_hard, penalty):
        """DifferentRoom: 课程必须在不同教室"""
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            rooms1 = self.class_to_room_options[c1]
            rooms2 = self.class_to_room_options[c2]
            
            common_rooms = set(rooms1) & set(rooms2)
            
            for r in common_rooms:
                if is_hard:
                    self.model.addConstr(
                        self.w[c1, r] + self.w[c2, r] <= 1,
                        name=f"diff_room_{c1}_{c2}_{r}"
                    )
                else:
                    p = self.model.addVar(vtype=GRB.BINARY)
                    self.model.addConstr(
                        self.w[c1, r] + self.w[c2, r] - 1 <= p
                    )
                    self.penalty_vars.append((p, penalty))
    
    def _add_not_overlap_constraint(self, classes, is_hard, penalty):
        """NotOverlap: 课程时间不能重叠"""
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    # ITC 2019: back-to-back (end1 == start2) is also a conflict
                    # for NotOverlap — students cannot be in two places at once
                    # even if the times are exactly adjacent.  Use strict < so
                    # back-to-back does NOT satisfy the "no conflict" condition.
                    if (start1 + end1 < start2) or (start2 + end2 < start1) or (and_days == 0) or (and_week == 0):
                        continue
                    else:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))
    
    def _add_overlap_constraint(self, classes, is_hard, penalty):
        """
        Overlap: 课程必须重叠
        即：至少有一对课程的时间必须有重叠
        
        这个约束的逻辑是：
        - 如果选择了classes中的任意两个课程，它们的时间必须重叠
        - 实现方式：对每对课程，如果它们的时间不重叠，则不能同时选择
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                        continue
                    else:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_same_attendees_constraint(self, classes, is_hard, penalty):
        """
        SameAttendees: 参与者可以参加所有课程
        即：课程时间不能重叠，且考虑教室间的旅行时间
        
        这个约束分为两部分：
        1. 时间重叠检查（已被NotOverlap覆盖）
        2. 时间-教室重叠检查（考虑旅行时间）
        """
        if len(classes) < 2:
            return
        
        # 获取travel时间矩阵（如果存在）
        travel_times = self.reader.travel if self.reader.travel else {}
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            room_opts1 = self.class_to_room_options[c1]
            room_opts2 = self.class_to_room_options[c2]
            
            for topt1, tidx1 in time_opts1:
                week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                
                for topt2, tidx2 in time_opts2:
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']

                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2

                    # Overlap — ITC 2019: back-to-back (end1==start2) counts as
                    # conflicting for SameAttendees (students need ≥1 slot gap).
                    # Use <= so that adjacent slots also trigger the constraint.
                    if (start1 <= start2 + end2) and (start2 <= start1 + end1) and (not and_days == 0) and (not and_week == 0):
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1,
                                name=f"same_att_time_{c1}_{c2}_{tidx1}_{tidx2}"
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))
                    # 检查时间-教室冲突（考虑旅行时间）
                    else:
                        # 如果两个课程在同一天但时间紧邻，需要检查教室距离
                        for r1 in room_opts1:
                            for r2 in room_opts2:
                                if r1 == 'dummy' or r2 == 'dummy':
                                    continue
                                
                                # 获取旅行时间
                                travel1 = travel_times.get(r1, {}).get(r2, 0)
                                travel2 = travel_times.get(r2, {}).get(r1, 0)
                                
                                # 检查是否有足够的时间旅行
                                if (start1 + end1 + travel1 <= start2) or (start2 + end2 + travel2 <= start1) or (and_days == 0) or (and_week == 0):
                                    continue
                                else:
                                    if is_hard:
                                        if (c1, tidx1, r1) in self.x and (c2, tidx2, r2) in self.x:
                                            self.model.addConstr(
                                                self.x[c1, tidx1, r1] + self.x[c2, tidx2, r2] <= 1,
                                                name=f"same_att_travel_{c1}_{tidx1}_{r1}_{c2}_{tidx2}_{r2}"
                                            )
                                    else:
                                        if (c1, tidx1, r1) in self.x and (c2, tidx2, r2) in self.x:
                                            p = self.model.addVar(vtype=GRB.BINARY)
                                            self.model.addConstr(
                                                self.x[c1, tidx1, r1] + self.x[c2, tidx2, r2] - 1 <= p
                                            )
                                            self.penalty_vars.append((p, penalty))

    def _add_precedence_constraint(self, classes, is_hard, penalty):
        """Precedence: 课程必须按顺序进行"""
        if len(classes) < 2:
            return
        
        # 假设classes列表的顺序就是优先级顺序
        for i in range(len(classes) - 1):
            c1 = classes[i]
            c2 = classes[i + 1]
            
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            # c1必须在c2之前结束
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                for topt2, tidx2 in time_opts2:
                    week_bits1, day_bits1, start1, end1 = topt1['optional_time_bits']
                    week_bits2, day_bits2, start2, end2 = topt2['optional_time_bits']
                    first_day1 = day_bits1.find('1')
                    first_day2 = day_bits2.find('1')
                    first_week1 = week_bits1.find('1')
                    first_week2 = week_bits2.find('1')
                    w_pre, d_pre, s_pre, e_pre = first_week1, first_day1, start1, end1
                    w_sub, d_sub, s_sub, e_sub = first_week2, first_day2, start2, end2
                    if (w_pre < w_sub) or ( # first(week_i) < first(week_j) or
                        (w_pre == w_sub) and (
                            (d_pre < d_sub ) or ( # first(day_i) < first(day_j) or
                                (d_pre == d_sub) and (s_pre + e_pre <= s_sub) # end_i <= start_j
                            )
                        )
                    ):
                        continue
                    else:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))
    
    def _add_workday_constraint(self, classes, max_slots, is_hard, penalty):
        """
        Workday(S): 限制每天的工作时长 - 优化版本
        
        使用枚举所有可能的时间跨度组合的方法
        这种方法更直接，但可能产生较多约束
        """
        if len(classes) == 0:
            return
        
        # 对每个week和day
        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                # 收集该天所有可能的课程时间段
                day_events = []
                
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        
                        if (w < len(weeks_bits) and weeks_bits[w] == '1' and
                            d < len(days_bits) and days_bits[d] == '1'):
                            end = start + length
                            day_events.append((cid, tidx, start, end))
                
                if len(day_events) == 0:
                    continue
                
                # 对于该天的所有课程组合，检查workday长度
                # 为了效率，我们只检查可能违反约束的组合
                
                # 如果只有一个课程，直接检查
                if len(day_events) == 1:
                    cid, tidx, start, end = day_events[0]
                    workday_length = end - start
                    
                    if workday_length > max_slots:
                        if is_hard:
                            # 这个课程不能被选择
                            self.model.addConstr(
                                self.y[cid, tidx] == 0,
                                name=f"workday_single_w{w}_d{d}_{cid}_{tidx}"
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(self.y[cid, tidx] <= p)
                            self.penalty_vars.append((p, penalty * (workday_length - max_slots)))
                    continue
                
                # 对于多个课程，检查所有可能的组合
                # 为了效率，我们使用启发式方法：
                # 找出可能导致workday超长的课程对
                
                for i in range(len(day_events)):
                    for j in range(i + 1, len(day_events)):
                        c1, t1, start1, end1 = day_events[i]
                        c2, t2, start2, end2 = day_events[j]
                        
                        # 计算如果这两个课程都被选中，workday的长度
                        earliest_start = min(start1, start2)
                        latest_end = max(end1, end2)
                        workday_length = latest_end - earliest_start
                        
                        # 如果超过限制，添加约束
                        if workday_length > max_slots:
                            if is_hard:
                                # 这两个课程不能同时被选中
                                self.model.addConstr(
                                    self.y[c1, t1] + self.y[c2, t2] <= 1,
                                    name=f"workday_pair_w{w}_d{d}_{c1}_{t1}_{c2}_{t2}"
                                )
                            else:
                                p = self.model.addVar(vtype=GRB.BINARY)
                                self.model.addConstr(
                                    self.y[c1, t1] + self.y[c2, t2] - 1 <= p
                                )
                                self.penalty_vars.append((p, penalty))
                
                # 对于3个或更多课程的组合
                # 这里我们只检查所有课程都被选中的情况
                if len(day_events) >= 3:
                    # 计算所有课程的总跨度
                    all_starts = [start for _, _, start, _ in day_events]
                    all_ends = [end for _, _, _, end in day_events]
                    total_span = max(all_ends) - min(all_starts)
                    
                    if total_span > max_slots:
                        if is_hard:
                            # 不能所有课程都被选中
                            self.model.addConstr(
                                gp.quicksum(self.y[c, t] for c, t, _, _ in day_events) 
                                <= len(day_events) - 1,
                                name=f"workday_all_w{w}_d{d}"
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY)
                            self.model.addConstr(
                                gp.quicksum(self.y[c, t] for c, t, _, _ in day_events)
                                - len(day_events) + 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_max_days_constraint(self, classes, max_days, is_hard, penalty):
        """MaxDays: 课程最多使用max_days天"""
        if len(classes) == 0:
            return
        
        # 创建辅助变量：gamma[d] = 1 if 任何课程在第d天
        gamma = {}
        for d in range(self.reader.nrDays):
            gamma[d] = self.model.addVar(vtype=GRB.BINARY, name=f"gamma_day_{d}")
        
        # 如果任何课程在第d天，则gamma[d] = 1
        M = len(classes)
        for d in range(self.reader.nrDays):
            day_vars = []
            for cid in classes:
                if cid not in self.reader.classes:
                    continue
                time_opts = self.class_to_time_options[cid]
                for topt, tidx in time_opts:
                    days_bits = topt['optional_time_bits'][1]
                    if d < len(days_bits) and days_bits[d] == '1':
                        day_vars.append(self.y[cid, tidx])
            
            if day_vars:
                self.model.addConstr(
                    gp.quicksum(day_vars) <= M * gamma[d]
                )
        
        # 总天数限制
        if is_hard:
            self.model.addConstr(
                gp.quicksum(gamma[d] for d in range(self.reader.nrDays)) <= max_days
            )
        else:
            p = self.model.addVar(vtype=GRB.INTEGER, lb=0, name="p_max_days")
            self.model.addConstr(
                gp.quicksum(gamma[d] for d in range(self.reader.nrDays)) - max_days <= p
            )
            self.penalty_vars.append((p, penalty))
    
    def _add_max_day_load_constraint(self, classes, max_slots, is_hard, penalty):
        """MaxDayLoad: 每天最多max_slots个时间槽"""
        if len(classes) == 0:
            return
        
        # 对每个week和每个day
        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                # 计算该天的总时长
                day_load = []
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        # 检查是否在这个week和day
                        if (w < len(weeks_bits) and weeks_bits[w] == '1' and
                            d < len(days_bits) and days_bits[d] == '1'):
                            day_load.append(length * self.y[cid, tidx])
                if day_load:
                    if is_hard:
                        self.model.addConstr(
                            gp.quicksum(day_load) <= max_slots,
                            name=f"max_day_load_w{w}_d{d}"
                        )
                    else:
                        p = self.model.addVar(vtype=GRB.INTEGER, lb=0)
                        self.model.addConstr(
                            gp.quicksum(day_load) - max_slots <= p
                        )
                        self.penalty_vars.append((p, penalty))

    def _add_min_gap_constraint(self, classes, min_gap, is_hard, penalty):
        """
        MinGap(G): 两个课程之间必须有至少G个时间槽的间隙
        """
        if len(classes) < 2:
            return
        
        for c1, c2 in combinations(classes, 2):
            if c1 not in self.reader.classes or c2 not in self.reader.classes:
                continue
            
            time_opts1 = self.class_to_time_options[c1]
            time_opts2 = self.class_to_time_options[c2]
            
            for topt1, tidx1 in time_opts1:
                bits1 = topt1['optional_time_bits']
                weeks1, days1, start1, length1 = bits1
                end1 = start1 + length1
                
                for topt2, tidx2 in time_opts2:
                    bits2 = topt2['optional_time_bits']
                    weeks2, days2, start2, length2 = bits2
                    end2 = start2 + length2
                    
                    # 检查是否在同一week和day
                    weeks_int1 = int(weeks1, 2)
                    weeks_int2 = int(weeks2, 2)
                    days_int1 = int(days1, 2)
                    days_int2 = int(days2, 2)
                    
                    has_common_time = ((weeks_int1 & weeks_int2) != 0 and 
                                    (days_int1 & days_int2) != 0)
                    
                    if not has_common_time:
                        continue
                    
                    # 计算实际间隙
                    if start2 >= end1:
                        gap = start2 - end1
                    elif start1 >= end2:
                        gap = start1 - end2
                    else:
                        gap = -1  # 重叠
                    
                    # 如果间隙小于最小要求，则冲突
                    if gap < min_gap:
                        if is_hard:
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] <= 1,
                                name=f"min_gap_{c1}_{c2}_{tidx1}_{tidx2}"
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.BINARY,
                                                name=f"p_min_gap_{c1}_{c2}_{tidx1}_{tidx2}")
                            self.model.addConstr(
                                self.y[c1, tidx1] + self.y[c2, tidx2] - 1 <= p
                            )
                            self.penalty_vars.append((p, penalty))

    def _add_max_breaks_constraint(self, classes, max_breaks, min_break_length, is_hard, penalty):
        # TODO
        """
        MaxBreaks(R,S): 限制课程间休息的次数
        R: 最大休息次数  
        S: 大于S个时间槽的间隙才算作休息
        
        核心思路：breaks数 = (合并后的块数) - 1
        其中合并时，gap ≤ S 的课程在同一块
        """
        if len(classes) == 0:
            return
        
        violation_cache = set()

        for w in range(self.reader.nrWeeks):
            for d in range(self.reader.nrDays):
                # 收集该天所有可能的课程
                day_events = []
                
                # bug1: timesolts of the same class append to day_events
                for cid in classes:
                    if cid not in self.reader.classes:
                        continue
                    
                    cls_events = []
                    time_opts = self.class_to_time_options[cid]
                    for topt, tidx in time_opts:
                        bits = topt['optional_time_bits']
                        weeks_bits, days_bits, start, length = bits
                        
                        if weeks_bits[w] == '1' and days_bits[d] == '1':
                            cls_events.append((cid, tidx, start, length))
                    if len(cls_events) > 0:
                        day_events.append(cls_events)
                
                for events in list(product(*day_events)):
                    events_cids_tidx = tuple([f"{event[0]} {event[1]}" for event in events])
                    if events_cids_tidx in violation_cache: continue
                    violation_cache.add(events_cids_tidx)
                    # 至少需要 R+2 门课才可能违反约束（因为 R+2 门课最少形成 R+1 个breaks）
                    if len(events) <= max_breaks + 1:
                        continue
                    
                    # 按开始时间排序
                    events = list(events)
                    events.sort(key=lambda x: x[2])
                    
                    # 创建块变量：表示是否开启一个新块
                    # new_block[i] = 1 表示第i门课开启了一个新块（即与前一门课的gap > S）
                    new_block_vars = []
                    
                    for i in range(1, len(events)):
                        c_prev, t_prev, start_prev, len_prev = events[i-1]
                        c_curr, t_curr, start_curr, len_curr = events[i]
                        
                        end_prev = start_prev + len_prev
                        gap = start_curr - end_prev
                        
                        if gap > min_break_length:
                            # 这个gap可能形成break
                            nb = self.model.addVar(vtype=GRB.BINARY,
                                                name=f"new_block_w{w}_d{d}_{i}")
                            
                            # nb = 1 当且仅当当前课程和前一课程都被选中
                            self.model.addConstr(nb <= self.y[c_curr, t_curr])
                            self.model.addConstr(nb <= self.y[c_prev, t_prev])
                            self.model.addConstr(
                                self.y[c_curr, t_curr] + self.y[c_prev, t_prev] - 1 <= nb
                            )
                            
                            new_block_vars.append(nb)
                    
                    # breaks数 = sum(new_block_vars)
                    # 要求 breaks数 ≤ max_breaks
                    if new_block_vars:
                        if is_hard:
                            self.model.addConstr(
                                gp.quicksum(new_block_vars) <= max_breaks,
                                name=f"max_breaks_w{w}_d{d}"
                            )
                        else:
                            p = self.model.addVar(vtype=GRB.INTEGER, lb=0,
                                                name=f"p_max_breaks_w{w}_d{d}")
                            self.model.addConstr(
                                gp.quicksum(new_block_vars) - max_breaks <= p,
                                name=f"max_breaks_soft_w{w}_d{d}"
                            )
                            self.penalty_vars.append((p, penalty))

    def merge_slots(self, class_time_slots, S):
        merge_time_slots = []
        merge_time_len = []
        breaks = -1
        class_time_slots = sorted(class_time_slots, key=lambda x: x[0])
        max_slots = 0
        for i, time_slot in enumerate(class_time_slots):
            if breaks == -1:
                merge_time_slots.append(time_slot)
                merge_time_len.append(1)
                breaks += 1
                continue
            start1, end1 = merge_time_slots[breaks]
            start2, end2 = time_slot
            if start1 + end1 + S >= start2:
                merge_time_slots[breaks][1] = max(start2 + end2, start1 + end1) - start1
                merge_time_len[breaks] += 1 
            else:
                merge_time_slots.append(time_slot)
                merge_time_len.append(1)
                breaks += 1
        return breaks, merge_time_slots, merge_time_len
    
    # def _add_max_breaks_constraint(self, classes, max_breaks, min_break_length, is_hard, penalty):
    #     """
    #     MaxBreaks(R,S): 限制课程间休息的次数
    #     R: 最大休息次数
    #     S: 大于S个时间槽的间隙才算作休息
        
    #     这是一个复杂的约束，需要对每个week和day进行检查
    #     """
    #     if len(classes) == 0:
    #         return
        
    #     # 对每个week和day
    #     for w in range(self.reader.nrWeeks):
    #         for d in range(self.reader.nrDays):
    #             # 收集该天所有可能的课程时间段
    #             day_events = []  # [(cid, tidx, start, end), ...]
                
    #             for cid in classes:
    #                 if cid not in self.reader.classes:
    #                     continue
                    
    #                 time_opts = self.class_to_time_options[cid]
    #                 for topt, tidx in time_opts:
    #                     bits = topt['optional_time_bits']
    #                     weeks_bits, days_bits, start, length = bits
                        
    #                     # 检查是否在这个week和day
    #                     if (w < len(weeks_bits) and weeks_bits[w] == '1' and
    #                         d < len(days_bits) and days_bits[d] == '1'):
    #                         day_events.append((cid, tidx, start, start + length))
                
    #             if len(day_events) < 2:
    #                 continue
                
    #             # 按开始时间排序
    #             day_events.sort(key=lambda x: x[2])
                
    #             # 创建辅助变量：表示每个可能的间隙是否是休息
    #             break_vars = []
                
    #             for i in range(len(day_events) - 1):
    #                 c1, t1, start1, end1 = day_events[i]
    #                 c2, t2, start2, end2 = day_events[i + 1]
                    
    #                 # 计算潜在的间隙
    #                 gap = start2 - end1
                    
    #                 # 如果间隙大于S，这可能是一个休息
    #                 if gap > min_break_length:
    #                     # 创建辅助变量：这个间隙是否是休息
    #                     b = self.model.addVar(vtype=GRB.BINARY,
    #                                         name=f"break_w{w}_d{d}_{i}")
                        
    #                     # 如果两个课程都被选中，则这是一个休息
    #                     self.model.addConstr(
    #                         self.y[c1, t1] + self.y[c2, t2] - 1 <= b,
    #                         name=f"break_def_w{w}_d{d}_{i}_a"
    #                     )
    #                     self.model.addConstr(
    #                         b <= self.y[c1, t1],
    #                         name=f"break_def_w{w}_d{d}_{i}_b"
    #                     )
    #                     self.model.addConstr(
    #                         b <= self.y[c2, t2],
    #                         name=f"break_def_w{w}_d{d}_{i}_c"
    #                     )
                        
    #                     break_vars.append(b)
                
    #             # 限制休息次数
    #             if break_vars:
    #                 if is_hard:
    #                     self.model.addConstr(
    #                         gp.quicksum(break_vars) <= max_breaks,
    #                         name=f"max_breaks_w{w}_d{d}"
    #                     )
    #                 else:
    #                     p = self.model.addVar(vtype=GRB.INTEGER, lb=0,
    #                                         name=f"p_max_breaks_w{w}_d{d}")
    #                     self.model.addConstr(
    #                         gp.quicksum(break_vars) - max_breaks <= p
    #                     )
    #                     self.penalty_vars.append((p, penalty))

    # def _add_max_block_constraint(self, classes, max_block_length, max_gap_in_block, is_hard, penalty):
    #     """
    #     MaxBlock(M,S): 限制连续课程块的长度
    #     M: 块的最大长度（时间槽）
    #     S: 小于S个时间槽的间隙仍被视为在同一块中
        
    #     一个"块"是一组课程，它们之间的间隙都小于S
    #     """
    #     if len(classes) == 0:
    #         return

    #     each_day = {}
    #     invalid_cache = {}  # {(w, d): set of invalid course combinations}
    #     # 对每个week和day
    #     for w in range(self.reader.nrWeeks):
    #         for d in range(self.reader.nrDays):
    #             # 收集该天所有可能的课程时间段
    #             each_day[w, d] = {}
    #             invalid_cache[w, d] = set()  # 初始化该天的无效缓存

    #             for cid in classes:
    #                 if cid not in self.reader.classes:
    #                     continue
    #                 day_events = [None] # wisedom
    #                 time_opts = self.class_to_time_options[cid]
    #                 for topt, tidx in time_opts:
    #                     bits = topt['optional_time_bits']
    #                     weeks_bits, days_bits, start, length = bits
    #                     if weeks_bits[w] == '1' and days_bits[d] == '1':
    #                         # 存储: (cid, tidx, start, length)
    #                         day_events.append((cid, tidx, start, length))
    #                 each_day[w, d][cid] = day_events

    #     for w in range(self.reader.nrWeeks):
    #         for d in range(self.reader.nrDays):
    #             valid_tops = product(*each_day[w, d].values())
    #             for idx, valid_top in enumerate(valid_tops):
    #                 if is_hard:
    #                     # 快速过滤：检查是否包含已知的无效课程组合
    #                     course_ids = tuple(sorted([each[0] for each in valid_top if each is not None]))
    #                     # 如果这个组合的任何子集在无效缓存中，跳过
    #                     skip = False
    #                     for invalid_combo in invalid_cache[w, d]:
    #                         if invalid_combo.issubset(set(course_ids)):
    #                             skip = True
    #                             break
    #                     if skip:
    #                         continue

    #                 y = []
    #                 valid_top_filtered = []
    #                 for each in valid_top:
    #                     if each == None:
    #                         continue
    #                     y.append((each[0], each[1]))
    #                     valid_top_filtered.append([each[2], each[3]])
                    
    #                 if len(valid_top_filtered) == 0: 
    #                     continue

    #                 overM_blocks = 0 # 超限块数
    #                 _, merge_time_slots, merge_time_len = self.merge_slots(valid_top_filtered, max_gap_in_block)
                    
    #                 for slots_len, slots in zip(merge_time_len, merge_time_slots):
    #                     if slots_len > 1 and slots[1] > max_block_length:
    #                         overM_blocks += 1
                    
    #                 if overM_blocks > 0:
    #                     if is_hard:
    #                         # 将这个组合添加到无效缓存
    #                         invalid_cache[w, d].add(frozenset(course_ids))
    #                         self.model.addConstr(
    #                             gp.quicksum(self.y[c, t] for c, t in y) 
    #                             <= len(y) - 1,
    #                             name=f"max_block_w{w}_d{d}_b{idx}"
    #                         )
    #                     else:
    #                         p = self.model.addVar(vtype=GRB.BINARY,
    #                                         name=f"p_max_block_w{w}_d{d}_b{idx}")
    #                         self.model.addConstr(
    #                             gp.quicksum(self.y[c, t] for c, t in y)
    #                             - len(y) + 1 <= len(y) * p,
    #                             name=f"max_block_soft_w{w}_d{d}_b{idx}"
    #                         )
    #                         self.penalty_vars.append((p, penalty * int(overM_blocks / max(self.reader.nrWeeks, 1))))

    # def _add_max_block_constraint(self, classes, max_block_length, max_gap_in_block, is_hard, penalty):
    #     """
    #     MaxBlock(M,S): 限制连续课程块的长度
    #     M: 块的最大长度（时间槽）
    #     S: 小于S个时间槽的间隙仍被视为在同一块中
        
    #     一个"块"是一组课程，它们之间的间隙都小于S
    #     """
    #     if len(classes) == 0:
    #         return
        
    #     # 对每个week和day
    #     for w in range(self.reader.nrWeeks):
    #         for d in range(self.reader.nrDays):
    #             # 收集该天所有可能的课程时间段
    #             day_events = []
                
    #             for cid in classes:
    #                 if cid not in self.reader.classes:
    #                     continue
                    
    #                 time_opts = self.class_to_time_options[cid]
    #                 for topt, tidx in time_opts:
    #                     bits = topt['optional_time_bits']
    #                     weeks_bits, days_bits, start, length = bits
                        
    #                     if (w < len(weeks_bits) and weeks_bits[w] == '1' and
    #                         d < len(days_bits) and days_bits[d] == '1'):
    #                         day_events.append((cid, tidx, start, start + length, length))
                
    #             if len(day_events) < 2:
    #                 continue
                
    #             # 按开始时间排序
    #             day_events.sort(key=lambda x: x[2])
                
    #             # 检查所有可能的块
    #             # 使用滑动窗口方法找到可能违反约束的块
    #             for i in range(len(day_events)):
    #                 # 尝试从第i个事件开始构建块
    #                 block_events = [day_events[i]]
    #                 block_start = day_events[i][2]
    #                 block_end = day_events[i][3]
                    
    #                 for j in range(i + 1, len(day_events)):
    #                     c_next, t_next, start_next, end_next, len_next = day_events[j]
                        
    #                     # 检查间隙
    #                     gap = start_next - block_end
                        
    #                     # 如果间隙小于S，这个事件属于当前块
    #                     if gap < max_gap_in_block:
    #                         block_events.append(day_events[j])
    #                         block_end = max(block_end, end_next)
    #                     else:
    #                         break
                    
    #                 # 检查块的总长度
    #                 block_length = block_end - block_start
                    
    #                 if block_length > max_block_length and len(block_events) > 1:
    #                     # 这个块违反约束
    #                     # 这些课程不能同时被选中
    #                     if is_hard:
    #                         self.model.addConstr(
    #                             gp.quicksum(self.y[c, t] for c, t, _, _, _ in block_events) 
    #                             <= len(block_events) - 1,
    #                             name=f"max_block_w{w}_d{d}_start{i}"
    #                         )
    #                     else:
    #                         p = self.model.addVar(vtype=GRB.BINARY,
    #                                             name=f"p_max_block_w{w}_d{d}_start{i}")
    #                         self.model.addConstr(
    #                             gp.quicksum(self.y[c, t] for c, t, _, _, _ in block_events)
    #                             - len(block_events) + 1 <= p
    #                         )
    #                         self.penalty_vars.append((p, penalty))

    def _find_max_block_minimal_violations(self, events, M, S):
        """
        MaxBlock(M,S) 核心算法：找所有"极小违规集"。

        输入:
            events: 当天所有课程时间段实例，已按 start 排序。
                    每项为 (cid, tidx, start, length)。
            M: 块最大长度（时间槽数）
            S: 合并间隙阈值（gap <= S 则视为同一块）

        返回:
            list of tuple(int,...)：每项是 events 的下标元组，
            代表一个极小违规集（这些事件同时选中会导致块超长，
            且去掉任意一个就不再违规）。

        正确性：
          Sound & Complete。极小违规集中的元素按 start 排序后必然形成
          "可串联链"（相邻 gap <= S）。DFS 沿可串联方向扩展，
          并允许跳过同 cid 的互斥事件，因此能遍历所有此类链。

        复杂度：O(E^2 * K) per (w,d)，E = 当天事件实例总数，
                K = 极小违规集的平均大小（极小性验证代价）。
                实际远好于此，因时间窗口和 cid 互斥大量剪枝。
        """
        n = len(events)
        all_violations = []
        seen = set()

        def _check_viol(indices):
            slots = [[events[k][2], events[k][3]] for k in indices]
            _, merge_slots_res, merge_lens = self.merge_slots(slots, S)
            for blen, bslot in zip(merge_lens, merge_slots_res):
                if blen > 1 and bslot[1] > M:
                    return True
            return False

        for start_idx in range(n):
            e0 = events[start_idx]
            # 栈: (chain, block_end, used_cids, search_from)
            stack = [([start_idx], e0[2] + e0[3], {e0[0]}, start_idx + 1)]

            while stack:
                chain, block_end, used_cids, search_from = stack.pop()
                block_start = events[chain[0]][2]

                for j in range(search_from, n):
                    ej_start = events[j][2]
                    ej_len   = events[j][3]
                    ej_cid   = events[j][0]

                    # 超出可串联范围，后续只会更晚，直接剪枝
                    if ej_start > block_end + S:
                        break

                    # 同 cid 互斥，跳过（不中断搜索，因为后面可能有其他 cid）
                    if ej_cid in used_cids:
                        continue

                    new_block_end = max(block_end, ej_start + ej_len)
                    new_span = new_block_end - block_start

                    if new_span > M:
                        # 候选违规集
                        candidate = tuple(sorted(chain + [j]))
                        if candidate not in seen:
                            seen.add(candidate)
                            # 验证极小性：去掉任意一个后是否仍违规
                            is_minimal = True
                            for k in range(len(candidate)):
                                sub = [candidate[l] for l in range(len(candidate)) if l != k]
                                if _check_viol(sub):
                                    is_minimal = False
                                    break
                            if is_minimal:
                                all_violations.append(candidate)
                        # 继续搜索（后续可能有新的 cid 形成其他极小违规集）
                    else:
                        # 未违规，继续扩展链
                        stack.append((
                            chain + [j],
                            new_block_end,
                            used_cids | {ej_cid},
                            j + 1
                        ))

        return all_violations

    def _add_max_block_constraint(self, classes, max_block_length, max_gap_in_block, is_hard, penalty):
        """
        MaxBlock(M,S): 限制连续课程块的长度。
        M: 块的最大长度（时间槽）
        S: gap <= S 的相邻课程视为同一块

        重构：用语义完整的极小违规集算法替代全组合枚举。
          旧方案复杂度：O(W x D x prod|Ti|)  —— 指数爆炸
          新方案复杂度：O(W x D x E^2)        —— 多项式，E = 当天事件实例总数
        """
        if len(classes) == 0:
            return

        # Step 1：预计算每个 cid 在每个 (w,d) 上的候选事件
        class_day_events = {}
        active_wd = set()

        for cid in classes:
            if cid not in self.reader.classes:
                continue
            if cid not in self.class_to_time_options:
                continue
            class_day_events[cid] = {}
            for topt, tidx in self.class_to_time_options[cid]:
                weeks_bits, days_bits, start, length = topt['optional_time_bits']
                for w, wb in enumerate(weeks_bits):
                    if wb != '1':
                        continue
                    for d, db in enumerate(days_bits):
                        if db != '1':
                            continue
                        key = (w, d)
                        class_day_events[cid].setdefault(key, []).append(
                            (tidx, start, length)
                        )
                        active_wd.add(key)

        # Step 2：对每个活跃的 (w,d) 找极小违规集并生成约束
        constr_count = 0

        for w, d in active_wd:
            day_events = []
            for cid in classes:
                if cid not in class_day_events:
                    continue
                for tidx, start, length in class_day_events[cid].get((w, d), []):
                    day_events.append((cid, tidx, start, length))

            if len(day_events) < 2:
                continue

            day_events.sort(key=lambda x: x[2])

            violations = self._find_max_block_minimal_violations(
                day_events, max_block_length, max_gap_in_block
            )

            for viol_indices in violations:
                y_vars = [
                    self.y[day_events[k][0], day_events[k][1]]
                    for k in viol_indices
                    if (day_events[k][0], day_events[k][1]) in self.y
                ]

                if len(y_vars) < 2:
                    continue

                if is_hard:
                    self.model.addConstr(
                        gp.quicksum(y_vars) <= len(y_vars) - 1,
                        name=f"maxblock_h_w{w}_d{d}_c{constr_count}"
                    )
                else:
                    # 计算该违规集对应的超限块数，用于 penalty 加权
                    slots = [[day_events[k][2], day_events[k][3]] for k in viol_indices]
                    _, merge_res, merge_lens = self.merge_slots(slots, max_gap_in_block)
                    over_blocks = sum(
                        1 for blen, bslot in zip(merge_lens, merge_res)
                        if blen > 1 and bslot[1] > max_block_length
                    )
                    scaled = penalty * int(over_blocks / max(self.reader.nrWeeks, 1))
                    effective_penalty = max(scaled, penalty)

                    p = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"maxblock_s_w{w}_d{d}_c{constr_count}"
                    )
                    self.model.addConstr(
                        gp.quicksum(y_vars) - len(y_vars) + 1 <= p,
                        name=f"maxblock_s_def_w{w}_d{d}_c{constr_count}"
                    )
                    self.penalty_vars.append((p, effective_penalty))

                constr_count += 1

    def _set_objective(self):
        """设置目标函数"""
        print("Setting objective...")
        
        active = self._active_classes

        # Priority 1: Minimize unassigned
        unassigned_obj = gp.quicksum(
            self.u[cid] for cid in self.reader.classes.keys()
            if active is None or cid in active
        )
        self.model.setObjectiveN(unassigned_obj, index=0, priority=10)

        obj_terms = []

        BIG_M = 100000

        # 1. 时间惩罚
        opt_weights = self.reader.optimization
        time_weight = opt_weights.get('time', 0) if opt_weights else 0

        for cid in self.reader.classes.keys():
            if active is not None and cid not in active:
                continue
            time_opts = self.class_to_time_options[cid]
            for topt, tidx in time_opts:
                penalty = topt.get('penalty', 0)
                if penalty > 0:
                    obj_terms.append(time_weight * penalty * self.y[cid, tidx])

        # 2. 教室惩罚
        room_weight = opt_weights.get('room', 0) if opt_weights else 0

        for cid in self.reader.classes.keys():
            if active is not None and cid not in active:
                continue
            class_data = self.reader.classes[cid]
            for ropt in class_data['room_options']:
                rid = ropt['id']
                penalty = ropt.get('penalty', 0)
                if penalty > 0 and (cid, rid) in self.w:
                    obj_terms.append(room_weight * penalty * self.w[cid, rid])
        
        # 3. 分布约束惩罚
        dist_weight = opt_weights.get('distribution', 0) if opt_weights else 0
        
        for p_var, cost in self.penalty_vars:
            obj_terms.append(dist_weight * cost * p_var)
        
        # 4. 未分配课程
        # obj_terms.append(gp.quicksum(self.u[cid] * BIG_M for cid in self.reader.classes.keys()))

        # 设置目标
        if obj_terms:
            self.model.setObjectiveN(
                gp.quicksum(obj_terms),
                index=1, priority=1
            )
            self.model.ModelSense = GRB.MINIMIZE
        else:
            # 如果没有惩罚，最小化总数（虚拟目标）
            self.model.setObjective(0, GRB.MINIMIZE)
        
        print(f"Objective set with {len(obj_terms)} terms")

    def solve(self):
        """求解模型"""
        print("\n=== Solving Model ===")
        start_time = time.time()
        
        self.model.optimize()
        
        solve_time = time.time() - start_time
        
        if self.model.Status == GRB.OPTIMAL or self.model.Status == GRB.SUBOPTIMAL:
            print(f"\n✓ Optimal solution found!")
            print(f"  Objective: {self.model.ObjVal:.2f}")
            print(f"  Time: {solve_time:.2f}s")
        elif self.model.Status == GRB.TIME_LIMIT:
            print(f"\n⚠ Time limit reached")
            if self.model.SolCount > 0:
                print(f"  Best objective: {self.model.ObjVal:.2f}")
                print(f"  Gap: {self.MIPGap*100:.2f}%")
        elif self.model.Status == GRB.INFEASIBLE:
            print(f"\n✗ Model is infeasible")
            self.model.computeIIS()
            self.model.write("infeasible.ilp")
            return None
        else:
            print(f"\n? Unknown status: {self.model.Status}")
            return None
        
        return self.extract_solution()
    
    def extract_solution(self):
        """提取解决方案"""
        if self.model.SolCount == 0:
            return None
        
        print("\n=== Extracting Solution ===")
        
        assignments_list = []
        for i in range(min(self.model.SolCount, 10)):
            assignments = {}
            self.model.setParam('SolutionNumber', i)
            ones = []
            zeros = []

            for (cid, tidx, rid), var in self.x.items():
                if self.x[cid, tidx, rid].Xn > 0.5:
                    ones.append(var)
                else:
                    zeros.append(var)

            self.model.addConstr(
                gp.quicksum(1 - v for v in ones) + 
                gp.quicksum(v for v in zeros) >= 1,
                name=f"exist_solution"
            )

            for cid in self.reader.classes.keys():
                # When running as a submodel, only the active classes are indexed
                if cid not in self.class_to_time_options:
                    continue
                class_data = self.reader.classes[cid]
                time_opts = self.class_to_time_options[cid]
                room_opts = self.class_to_room_options[cid]
                
                assigned_time = None
                assigned_room = None

                # 找到被选中的时间
                for topt, tidx in time_opts:
                    if (cid, tidx) in self.y and self.y[cid, tidx].Xn > 0.5:
                        assigned_time = topt
                        break
                
                # 找到被选中的教室
                for rid in room_opts:
                    if (cid, rid) in self.w and self.w[cid, rid].Xn > 0.5:
                        assigned_room = rid
                        break
                
                # 保存分配
                room_required = class_data.get('room_required', True)
                if assigned_room == 'dummy':
                    assigned_room = None
                
                assignments[cid] = (
                    assigned_time,
                    room_required,
                    assigned_room,
                    []  # student_ids (不实现学生分配)
                )
            assignments_list.append(assignments)
                # if assigned_time:
                #     bits = assigned_time['optional_time_bits']
                    # print(f"  Class {cid}: weeks={bits[0][:8]}... days={bits[1]} "
                    #     f"start={bits[2]} length={bits[3]} room={assigned_room}")
            self.logger.info(f"\nAssigned {len([a for a in assignments.values() if a[0] is not None])} / {len(assignments)} classes")
        
        return assignments_list

    def load_model(self, model_path):
        self.model = gp.read(f"{model_path}.mps")
        self.model.read(f"{model_path}.prm")
        self.logger.info(f"\n✓ Model load from: {model_path}.mps")
        self.logger.info(f"\n✓ Param load from: {model_path}.prm")

    def save_model(self, output_path):
        self.model.write(f"{output_path}.mps")
        self.model.write(f"{output_path}.prm")
        self.logger.info(f"\n✓ Model saved to: {output_path}.mps")
        self.logger.info(f"\n✓ Param saved to: {output_path}.prm")

    def save_solution(self, assignments, output_path, config):
        """保存解决方案到XML"""
        if assignments is None:
            print("No solution to save")
            return
        
        export_solution_xml(
            assignments=assignments,
            out_path=output_path,
            name=self.reader.problem_name,
            runtime_sec=self.model.Runtime,
            cores=self.Threads,
            technique=config['config']['technique'],
            author=config['config']['author'],
            institution=config['config']['institution'],
            country=config['config']['country'],
            include_students=config['config']['include_students']
        )
        
        self.logger.info(f"\n✓ Solution saved to: {output_path}")

        # solu = solution(self.reader, output_path)
        # result = solu.total_penalty()
        return None
    
    def reset(self):
        self.time_assignments = {}
        self.room_assignments = {}
        self.model.reset()

    # ------------------------------------------------------------------ #
    # Hybrid solver interface                                              #
    # ------------------------------------------------------------------ #

    def fix_assignments(self, fixed: dict):
        """
        Pin Layer-1 community assignments before solve().

        Parameters
        ----------
        fixed : {cid: (tidx, rid_or_None)}
            For each class in `fixed`, forces:
              - y[cid, tidx]       == 1
              - y[cid, other_tidx] == 0  for all other time options
              - x[cid, tidx, rid]  == 1  (if rid is not None)
              - u[cid]             == 0

        Must be called after build_model() and before solve().
        To undo, call reset_fixed().
        """
        if self._fix_constraints:
            self.logger.warning(
                "fix_assignments() called while previous fixed constraints still active. "
                "Call reset_fixed() first to avoid duplicate constraints."
            )

        n_vars_fixed = 0
        for cid, (tidx, rid) in fixed.items():
            self._fixed_assignments[cid] = (tidx, rid)

            # Force chosen time option ON
            if (cid, tidx) in self.y:
                name = f"fix_y_{cid}_{tidx}"
                c = self.model.addConstr(self.y[cid, tidx] == 1, name=name)
                self._fix_constraints.append(c)
                self._fix_constr_to_cid[name] = cid
                n_vars_fixed += 1

            # Force all other time options OFF
            for _topt, other_tidx in self.class_to_time_options.get(cid, []):
                if other_tidx != tidx and (cid, other_tidx) in self.y:
                    name = f"fix_y0_{cid}_{other_tidx}"
                    c = self.model.addConstr(self.y[cid, other_tidx] == 0, name=name)
                    self._fix_constraints.append(c)
                    self._fix_constr_to_cid[name] = cid

            # Force chosen room ON
            if rid is not None and (cid, tidx, rid) in self.x:
                name = f"fix_x_{cid}_{tidx}_{rid}"
                c = self.model.addConstr(self.x[cid, tidx, rid] == 1, name=name)
                self._fix_constraints.append(c)
                self._fix_constr_to_cid[name] = cid

            # Force unassigned variable OFF
            if cid in self.u:
                name = f"fix_u_{cid}"
                c = self.model.addConstr(self.u[cid] == 0, name=name)
                self._fix_constraints.append(c)
                self._fix_constr_to_cid[name] = cid

        self.model.update()
        self.logger.info(
            f"fix_assignments: pinned {len(fixed)} classes "
            f"({len(self._fix_constraints)} constraints added)"
        )

    def get_infeasibility_info(self) -> dict:
        """
        After solve() returns None due to INFEASIBLE, identify which fixed
        class assignments participate in the IIS.

        Requires solve() to have been called (it calls computeIIS() internally).
        The IIS is already computed and written to infeasible.ilp by solve().

        Returns
        -------
        {
            "implicated_cids"      : set[str]   — fixed class IDs in the IIS,
            "violated_constraints" : list[str]  — names of all IIS constraints,
        }
        """
        if self.model.Status != GRB.INFEASIBLE:
            self.logger.warning(
                "get_infeasibility_info() called on a non-infeasible model "
                f"(status={self.model.Status}). Returning empty result."
            )
            return {"implicated_cids": set(), "violated_constraints": []}

        # IIS is computed by solve(); collect names of all IIS constraints
        violated_constraint_names = [
            c.ConstrName
            for c in self.model.getConstrs()
            if c.IISConstr
        ]

        # Among those, find which belong to our fix constraints
        implicated_cids: set = set()
        for name in violated_constraint_names:
            if name in self._fix_constr_to_cid:
                implicated_cids.add(self._fix_constr_to_cid[name])

        self.logger.info(
            f"IIS: {len(violated_constraint_names)} constraints involved; "
            f"{len(implicated_cids)} fixed classes implicated: {implicated_cids}"
        )
        return {
            "implicated_cids": implicated_cids,
            "violated_constraints": violated_constraint_names,
        }

    def reset_fixed(self):
        """
        Remove all constraints added by fix_assignments(), restoring the model
        to its post-build_model() state so a new community assignment can be
        tested without rebuilding the entire model.
        """
        for c in self._fix_constraints:
            self.model.remove(c)
        self._fix_constraints.clear()
        self._fix_constr_to_cid.clear()
        self._fixed_assignments.clear()
        self.model.update()
        self.logger.info("reset_fixed: all fixed-assignment constraints removed")

    # ------------------------------------------------------------------ #
    # Divide-and-conquer submodel interface                               #
    # ------------------------------------------------------------------ #

    def build_submodel(
        self,
        class_ids: set,
        hard_constraints: list,
        soft_constraints: list,
    ) -> None:
        """
        Build a MIP model for a *subset* of classes (one partition).

        All existing build_model() logic is reused; the three ``_active_*``
        attributes act as filters so only the specified classes and their
        intra-partition constraints are added.

        Parameters
        ----------
        class_ids         : set of class IDs to include
        hard_constraints  : intra-partition hard constraints (pre-filtered)
        soft_constraints  : intra-partition soft constraints (pre-filtered)
        """
        self._active_classes = set(class_ids)
        self._active_hard_constraints = hard_constraints
        self._active_soft_constraints = soft_constraints
        try:
            self.build_model()
        finally:
            # Always clear the overrides so accidental reuse of the solver
            # for a full model still works correctly.
            self._active_classes = None
            self._active_hard_constraints = None
            self._active_soft_constraints = None

    def get_simple_assignments(self) -> dict:
        """
        Extract {cid: (tidx, rid_or_None)} from the most recently solved model.

        Returns an empty dict if no solution is available.
        """
        if self.model.SolCount == 0:
            return {}
        result = {}
        for (cid, tidx, rid), var in self.x.items():
            try:
                if var.X > 0.5:
                    result[cid] = (tidx, None if rid == 'dummy' else rid)
            except Exception:
                pass
        return result

    def forbid_time_room(
        self,
        cid: str,
        tidx: int,
        rid: str = None,
    ) -> bool:
        """
        Add a constraint that forbids class `cid` from using the given
        time option (and, optionally, the specific room).

        If `rid` is given and the x[cid,tidx,rid] variable exists, only
        that room-time combination is forbidden (the class can still use a
        different room at the same time).

        If `rid` is None (or the x variable doesn't exist), the entire time
        option y[cid,tidx] is forbidden.

        Returns True if a constraint was successfully added, False otherwise.
        """
        added = False

        if rid is not None and rid != 'dummy' and (cid, tidx, rid) in self.x:
            name = f"forbid_x_{cid}_{tidx}_{rid}"
            c = self.model.addConstr(self.x[cid, tidx, rid] == 0, name=name)
            self._forbid_constraints.append(c)
            added = True
        elif (cid, tidx) in self.y:
            name = f"forbid_y_{cid}_{tidx}"
            c = self.model.addConstr(self.y[cid, tidx] == 0, name=name)
            self._forbid_constraints.append(c)
            added = True

        if added:
            self.model.update()
        return added

    def reset_forbid(self) -> None:
        """Remove all constraints added by forbid_time_room()."""
        for c in self._forbid_constraints:
            self.model.remove(c)
        self._forbid_constraints.clear()
        self.model.update()

    def reserve_room_times(self, reserved_room_times: set) -> int:
        """
        Pre-block room-time combinations already claimed by earlier partitions.

        For every x[cid,tidx,rid] variable in this model, if (rid, time_bits)
        overlaps with any entry in ``reserved_room_times``, add x == 0.

        This enables conflict-free sequential solving: each sub-MIP is told
        which (room, time) slots are already occupied before it is solved, so
        room double-booking is impossible by construction.

        Parameters
        ----------
        reserved_room_times : set of (rid, time_bits) where
            time_bits = (weeks_bits_str, days_bits_str, start_int, length_int)

        Returns
        -------
        int — number of constraints added (informational)
        """
        if not reserved_room_times:
            return 0

        # Group by room for O(1) lookup per variable
        reserved_by_rid = defaultdict(list)
        for (rid, tbits) in reserved_room_times:
            reserved_by_rid[rid].append(tbits)

        added = 0
        for (cid, tidx, rid), var in self.x.items():
            if rid == 'dummy' or rid not in reserved_by_rid:
                continue
            tbits = self.reader.classes[cid]["time_options"][tidx]["optional_time_bits"]
            for reserved_bits in reserved_by_rid[rid]:
                if self._time_bits_conflict(tbits, reserved_bits):
                    name = f"reserve_{cid}_{tidx}_{rid}"
                    c = self.model.addConstr(var == 0, name=name)
                    self._forbid_constraints.append(c)
                    added += 1
                    break  # one reserved match is enough to block this var

        if added:
            self.model.update()
        return added

    def add_preassigned_time_constraints(
        self,
        preassigned: dict,
        constraints: list,
    ) -> int:
        """
        Enforce hard time constraints between classes in this sub-model and
        classes that have already been assigned in earlier partitions.

        For each constraint in ``constraints`` that mixes in-model classes with
        pre-assigned classes:

          NotOverlap / SameAttendees
              Block every time option for the in-model class that would overlap
              with the pre-assigned class's time.

          SameTime
              Block every time option that does NOT exactly match the
              pre-assigned class's time (week/day/start/length all equal).

        Other constraint types are silently skipped; they should be rare or
        handled by the intra-partition MIP directly.

        Parameters
        ----------
        preassigned : {cid: (tidx, rid)} assignments from already-solved partitions
        constraints : hard constraint dicts that span in-model and pre-assigned classes

        Returns
        -------
        int  — number of y[cid, tidx] variables blocked
        """
        blocked = 0
        blocked_y: set = set()   # (cid, tidx) already blocked — avoid duplicate constrs

        for cons in constraints:
            ctype  = cons.get("type", "")
            classes = cons["classes"]

            in_model  = [c for c in classes if c in self.class_to_time_options]
            pre_asgnd = [(c, preassigned[c]) for c in classes if c in preassigned]

            if not in_model or not pre_asgnd:
                continue

            if ctype in ("SameAttendees", "NotOverlap"):
                for (other_cid, (other_tidx, _)) in pre_asgnd:
                    if other_tidx is None:
                        continue
                    other_bits = (
                        self.reader.classes[other_cid]
                        ["time_options"][other_tidx]["optional_time_bits"]
                    )
                    for cid in in_model:
                        for (_, tidx) in self.class_to_time_options.get(cid, []):
                            if (cid, tidx) in blocked_y:
                                continue
                            my_bits = (
                                self.reader.classes[cid]
                                ["time_options"][tidx]["optional_time_bits"]
                            )
                            if self._time_bits_conflict(my_bits, other_bits, attendee=True):
                                if (cid, tidx) in self.y:
                                    c = self.model.addConstr(
                                        self.y[cid, tidx] == 0,
                                        name=f"cross_no_overlap_{cid}_{tidx}",
                                    )
                                    self._forbid_constraints.append(c)
                                    blocked_y.add((cid, tidx))
                                    blocked += 1

            elif ctype == "SameTime":
                for (other_cid, (other_tidx, _)) in pre_asgnd:
                    if other_tidx is None:
                        continue
                    other_bits = (
                        self.reader.classes[other_cid]
                        ["time_options"][other_tidx]["optional_time_bits"]
                    )
                    for cid in in_model:
                        for (_, tidx) in self.class_to_time_options.get(cid, []):
                            if (cid, tidx) in blocked_y:
                                continue
                            my_bits = (
                                self.reader.classes[cid]
                                ["time_options"][tidx]["optional_time_bits"]
                            )
                            # Block everything that does NOT exactly match
                            if my_bits != other_bits:
                                if (cid, tidx) in self.y:
                                    c = self.model.addConstr(
                                        self.y[cid, tidx] == 0,
                                        name=f"cross_same_time_{cid}_{tidx}",
                                    )
                                    self._forbid_constraints.append(c)
                                    blocked_y.add((cid, tidx))
                                    blocked += 1

        if blocked:
            self.model.update()
        return blocked

    @staticmethod
    def _time_bits_conflict(bits1: tuple, bits2: tuple, attendee: bool = False) -> bool:
        """Return True if two time_bits tuples represent conflicting schedules.

        Parameters
        ----------
        attendee : bool
            When False (default) uses strict overlap < (back-to-back is OK).
            Used for room reservation: two classes can use the same room
            consecutively without conflict.
            When True uses inclusive overlap <= (back-to-back is a conflict).
            Used for SameAttendees / NotOverlap cross-partition enforcement:
            students cannot attend two consecutive classes with no gap.
        """
        w1, d1, s1, l1 = bits1
        w2, d2, s2, l2 = bits2
        # Slot overlap
        if attendee:
            if not (s1 <= s2 + l2 and s2 <= s1 + l1):
                return False
        else:
            if not (s1 < s2 + l2 and s2 < s1 + l1):
                return False
        # Day overlap
        if (int(d1, 2) & int(d2, 2)) == 0:
            return False
        # Week overlap
        if (int(w1, 2) & int(w2, 2)) == 0:
            return False
        return True