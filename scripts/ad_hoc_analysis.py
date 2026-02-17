import pandas as pd


# first I want a barplot showing percentage of samples containing each taxon, for the top 20 most prevalent taxa. This will be a horizontal barplot with taxa on the y-axis and percentage of samples on the x-axis. I will also add a vertical line at 50% to show which taxa are present in more than half of the samples.
import matplotlib.pyplot as plt

def plot_prevalence_barplot(abundance_table: pd.DataFrame, top_n: int = 20) -> None:
    # Calculate prevalence for each taxon (assuming taxa are in columns and samples in rows)
    prevalence = (abundance_table > 0).sum(axis=1) / abundance_table.shape[1] * 100
    
    # Get the top N most prevalent taxa
    top_taxa = prevalence.sort_values(ascending=False).head(top_n)
    top_taxa.index.name = "Taxon"
    top_taxa.name = "Prevalence (%)"

    # split index names
    top_taxa.index = top_taxa.index.str.split(";c__").str[0]  # keep only the last part of the taxonomic string
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 6))
    top_taxa.sort_values().plot(kind='barh', color='skyblue')
    
    # Add vertical line at 50%
    plt.axvline(x=50, color='red', linestyle='--')
    
    plt.xlabel('Percentage of Samples Containing Taxon (%)')
    plt.title(f'Top {top_n} Most Prevalent Taxa')
    plt.tight_layout()
    plt.savefig(f"outputs/prevalence_barplot_top_{top_n}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Example usage
    abundance_table = pd.read_csv("outputs/abundance_full.csv", index_col=0)
    
    plot_prevalence_barplot(abundance_table)