import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from evaluation metrics.")
    parser.add_argument("--input", type=str, default="results/evaluation_summary.json", help="Path to the JSON summary file.")
    parser.add_argument("--output", type=str, default="results/latex_table.tex", help="Path to save the LaTeX table code.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: summary file {args.input} does not exist.")
        return
        
    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {args.input}: {e}")
        return
        
    # Sort or order by context_in and context_out to match the desired format
    order_in = ["user profile", "review history", "item-review history"]
    order_out = ["item title", "item title and description"]
    
    # Map raw context names to pretty names for LaTeX
    pretty_in = {
        "user profile": "User Profile",
        "review history": "Review History",
        "item-review history": "Item-Review History"
    }
    pretty_out = {
        "item title": "Title",
        "item title and description": "Title + Description"
    }
    
    rows = []
    # Fill rows based on predefined order
    for cin in order_in:
        for cout in order_out:
            # find matching entry in data
            match = None
            for entry in data:
                if entry.get("context_in") == cin and entry.get("context_out") == cout:
                    match = entry
                    break
            if match:
                rows.append(match)
            else:
                rows.append({
                    "context_in": cin,
                    "context_out": cout,
                    "rmse": None,
                    "mae": None,
                    "map": None,
                    "ndcg10": None
                })
                
    # Build LaTeX tabular content
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Evaluation Results for Different Context Configurations}")
    latex_lines.append(r"\label{tab:evaluation_results}")
    latex_lines.append(r"\begin{tabular}{llcccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\textbf{Context In} & \textbf{Context Out} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAP} & \textbf{NDCG@10} \\ \hline")
    
    for row in rows:
        cin_pretty = pretty_in.get(row["context_in"], row["context_in"])
        cout_pretty = pretty_out.get(row["context_out"], row["context_out"])
        
        mae_str = f"{row['mae']:.4f}" if row["mae"] is not None else "-"
        rmse_str = f"{row['rmse']:.4f}" if row["rmse"] is not None else "-"
        map_str = f"{row['map']:.4f}" if row["map"] is not None else "-"
        ndcg_str = f"{row['ndcg10']:.4f}" if row["ndcg10"] is not None else "-"
        
        latex_lines.append(f"{cin_pretty} & {cout_pretty} & {mae_str} & {rmse_str} & {map_str} & {ndcg_str} \\\\")
        
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    # Print to console
    print("\n--- Generated LaTeX Table ---")
    print(latex_code)
    print("-----------------------------\n")
    
    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table saved to {args.output}")

if __name__ == "__main__":
    main()
