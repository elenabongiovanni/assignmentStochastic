import os
import json

def save_newsvendor_results(val_mc, quantity_mc, demand_mc, val_mm, quantity_mm, dist_mm, demand_mm, val_w, quantity_w, dist_w, demand_w, filename="NV_results.txt"):
    path = os.path.join("result", filename)
    with open(path, "w") as f:
        f.write("=== NEWSVENDOR RESULTS ===\n")
        
        f.write(f"Optimal value with full MC: {val_mc:.4f}\n")
        f.write(f"Optimal quantity (MC): {quantity_mc:.2f}\n\n")

        f.write(f"Optimal value with Moment Matching: {val_mm:.4f}\n")
        f.write(f"Optimal quantity (Moment Matching): {quantity_mm:.2f}\n")
        f.write(f"Moment Matching distance: {dist_mm:.4f}\n\n")

        f.write(f"Optimal value with Wasserstein: {val_w:.4f}\n")
        f.write(f"Optimal quantity (Wasserstein): {quantity_w:.2f}\n")
        f.write(f"Wasserstein distance: {dist_w:.4f}\n\n")

        diff_mm = abs(val_mm - val_mc)
        diff_w = abs(val_w - val_mc)
        f.write(f"Difference MM - MC: {diff_mm:.4f}\n")
        f.write(f"Difference Wasserstein - MC: {diff_w:.4f}\n")

    print(f"NewsVendor results saved in {path}")

    results_nv_dict = {
        "val_mc": val_mc,
        "opt_mc": quantity_mc,
        "demand_mc" : demand_mc.tolist(),
        "val_mm": val_mm,
        "opt_mm": quantity_mm,
        "dist_mm": dist_mm,
        "demand_mm" : demand_mm.tolist(),
        "val_w": val_w,
        "opt_w": quantity_w,
        "dist_w": dist_w,
        "demand_w" : demand_w.tolist()
    }

    json_path = os.path.join("result", "nv_results.json")
    with open(json_path, "w") as jf:
        json.dump(results_nv_dict, jf, indent=4)

    return results_nv_dict


def save_newsvendor_results_stability(val_mc=None, val_mm=None, dist_mm=None, val_w=None, dist_w=None,
                                     in_sample=None, out_sample_std=None, out_sample_cv=None,
                                     history=None,
                                     filename_txt="NV_results.txt",
                                     filename_json="nv_results.json"):
    
    # Salvataggio su file di testo (come prima)
    path_txt = os.path.join("result", filename_txt)
    with open(path_txt, "a") as f:
        f.write("\n=== NEWSVENDOR STABILITY ===\n")
        if in_sample is not None:
            f.write(f"In-sample stability (min number of scenarios): {in_sample}\n")
        if out_sample_std is not None and out_sample_cv is not None:
            f.write(f"Out-of-sample standard deviation: {out_sample_std:.4f}\n")
            f.write(f"Out-of-sample coefficient of variation: {out_sample_cv:.4f}\n")
    
    print(f"NewsVendor stability saved in {path_txt}")

    # Salvataggio history in JSON, se presente
    if history is not None:
        path_json = os.path.join("result", "nv_results_stability.json")
        with open(path_json, "w") as jf:
            json.dump(history, jf, indent=4)
        print(f"NewsVendor stability history saved in {path_json}")


