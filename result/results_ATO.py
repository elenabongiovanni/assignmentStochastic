import os
import json

def save_ato_results(val_mc, opt_mc, demand_mc, val_mm, opt_mm, dist_mm, demand_mm, val_w, opt_w, dist_w, demando_w, filename="ATO_results.txt"):

    path = os.path.join("result", filename)

    with open(path, "w") as f:
        f.write("=== ATO RESULTS ===\n")
        f.write(f"Optimal value with full MC: {val_mc:.4f}\n")
        f.write("Produced quantities (Full MC): ")
        f.write(", ".join([f"{q:.2f}" for q in opt_mc]) + "\n\n")

        f.write(f"Optimal value with Moment Matching: {val_mm:.4f}\n")
        f.write("Produced quantities (Moment Matching): ")
        f.write(", ".join([f"{q:.2f}" for q in opt_mm]) + "\n")
        f.write(f"Moment Matching distance: {dist_mm:.4f}\n\n")

        f.write(f"Optimal value with Wasserstein: {val_w:.4f}\n")
        f.write("Produced quantities (Wasserstein): ")
        f.write(", ".join([f"{q:.2f}" for q in opt_w]) + "\n")
        f.write(f"Wasserstein distance: {dist_w:.4f}\n\n")

        diff_mm = abs(val_mm - val_mc)
        diff_w = abs(val_w - val_mc)
        f.write(f"Difference between MM - MC: {diff_mm:.4f}\n")
        f.write(f"Difference between Wasserstein - MC: {diff_w:.4f}\n")

    print(f"ATO results saved in {path}")

    results_ato_dict = {
        "val_mc": val_mc,
        "opt_mc": opt_mc,
        "demand_mc" : demand_mc.tolist(),
        "val_mm": val_mm,
        "opt_mm": opt_mm,
        "dist_mm": dist_mm,
        "demand_mm" : demand_mm.tolist(),
        "val_w": val_w,
        "opt_w": opt_w,
        "dist_w": dist_w,
        "demand_w" : demando_w.tolist()
    }

    json_path = os.path.join("result", "ato_results.json")
    with open(json_path, "w") as jf:
        json.dump(results_ato_dict, jf, indent=4)

    return results_ato_dict


def save_ato_results_stability(val_mc=None, val_mm=None, dist_mm=None, val_w=None, dist_w=None,
                               in_sample=None, out_sample_std=None, out_sample_cv=None,
                               history=None,
                               filename_txt="ATO_results.txt",
                               filename_json="ATO_results.json"):
    
    # Save results on the txt file
    path_txt = os.path.join("result", filename_txt)
    with open(path_txt, "a") as f:
        f.write("\n=== ATO STABILITY ===\n")
        if in_sample is not None:
            f.write(f"In-sample stability (min number of scenarios): {in_sample}\n")
        if out_sample_std is not None and out_sample_cv is not None:
            f.write(f"Out-of-sample standard deviation: {out_sample_std:.4f}\n")
            f.write(f"Out-of-sample coefficient of variation: {out_sample_cv:.4f}\n")
    
    print(f"ATO stability saved in {path_txt}")

    # Save results on the JSON file
    if history is not None:
        path_json = os.path.join("result", "ato_results_stability.json")
        with open(path_json, "w") as jf:
            json.dump(history, jf, indent=4)
        print(f"ATO stability saved in {path_json}")

