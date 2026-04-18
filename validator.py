import os
from src.utils.validator import solution
from src.utils.dataReader import PSTTReader

def solus_validate(pname, xml_path, solu_path):
    file = f'{xml_path}/{pname}.xml'
    reader = PSTTReader(file)
    solution_list = []
    invalid_solutions = []
    for solu_file in os.listdir(solu_path):
        if solu_file.endswith('.xml') and solu_file.startswith('solution'):
            valid, result = solu_validate(pname, xml_path, solu_path, solu_file, reader)
            if valid:
                solution_list.append((solu_file, result["Total_cost"], result["Time penalty"], result["Room penalty"], result["Distribution penalty"]))
            else:
                invalid_solutions.append(result)
    if len(solution_list) > 0:
        solution_list = sorted(solution_list, key=lambda k: k[1])
        print("The best solution: ")
        print("Solution: ", solution_list[0][0])
        print("Total_cost: ", solution_list[0][1])
        print("Time penalty: ", solution_list[0][2])
        print("Room penalty: ", solution_list[0][3])
        print("Distribution penalty: ", solution_list[0][4])
    else:
        not_assigned_count = {}
        for result in invalid_solutions:
            for cid in result['not assignment']:
                if not_assigned_count.get(cid, 0) == 0: not_assigned_count[cid] = 1
                else: not_assigned_count[cid] += 1
        print(not_assigned_count)
        print("relative hard constraints:")
        for cid in not_assigned_count.keys():
            for hc in reader.distributions['hard_constraints']:
                if cid in hc['classes']:
                    print(hc)

def solu_validate(pname, xml_path, solu_path, solu_file, reader=None):
    if reader == None: 
        file = f'{xml_path}/{pname}.xml'
        reader = PSTTReader(file)
    solution_file = f"{solu_path}/{solu_file}"
    solu = solution(reader, solution_file)
    result = solu.total_penalty()
    print()
    if result['valid']:
        print(f"valid solution: {solution_file}")
        print("Total_cost", result["Total_cost"])
        print("Time penalty", result["Time penalty"])
        print("Room penalty", result["Room penalty"])
        print("Distribution penalty", result["Distribution penalty"])
        return True, result
    else:
        print(f"invalid solution: {solution_file}")
        print(f"not assigned classes:", result['not assignment'])
        # exit(1)
    return False, result

if __name__ == "__main__":
    pname = 'lums-spr18'
    xml_path = "/home/scxsz1/zsh/FYP/STT_last_version/data/source/reduced"
    solu_path = f"/home/scxsz1/zsh/FYP/STT_last_version/solutions/{pname}"
    solus_validate(pname, xml_path, solu_path)

    # solu_file = "solution74_pu-llr-spr17.xml"
    # result = solu_validate(pname, xml_path, solu_path, solu_file)
    # Total_cost = result["Total_cost"]