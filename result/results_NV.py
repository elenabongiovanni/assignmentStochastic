import os

def save_newsvendor_results(val_mc, val_mm, dist_mm, val_w, dist_w, filename="NV_results.txt"):
    path = os.path.join("result", filename)
    with open(path, "w") as f:
        f.write("=== RISULTATI NEWSVENDOR ===\n")
        f.write(f"Valore ottimo con MC completo: {val_mc:.4f}\n")
        f.write(f"Valore ottimo con Moment Matching: {val_mm:.4f}\n")
        f.write(f"Distanza Moment Matching: {dist_mm:.4f}\n")
        f.write(f"Valore ottimo con Wasserstein: {val_w:.4f}\n")
        f.write(f"Distanza Wasserstein: {dist_w:.4f}\n")

        diff_mm = abs(val_mm - val_mc)
        diff_w = abs(val_w - val_mc)
        f.write(f"Differenza Moment Matching - MC: {diff_mm:.4f}\n")
        f.write(f"Differenza Wasserstein - MC: {diff_w:.4f}\n")

    print(f"[✓] Risultati NewsVendor salvati in {path}")

    import os

def save_newsvendor_results_stability(val_mc=None, val_mm=None, dist_mm=None, val_w=None, dist_w=None,
                            in_sample=None, out_sample_std=None, out_sample_cv=None,
                            filename="NV_results.txt"):
    path = os.path.join("result", filename)
    with open(path, "a") as f:
        f.write("\n=== STABILITA' NEWSVENDOR ===\n")
        if in_sample is not None:
            f.write(f"In-sample stability (min n_scenarios): {in_sample}\n")
        if out_sample_std is not None and out_sample_cv is not None:
            f.write(f"Out-of-sample STD: {out_sample_std:.4f}\n")
            f.write(f"Out-of-sample CV : {out_sample_cv:.4f}\n")
    print(f"[✓] Stabilità NewsVendor salvata in {path}")
