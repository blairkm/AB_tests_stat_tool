import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from itertools import combinations
from scipy.stats import chi2_contingency

class ABTestProportionsTool:
    def __init__(self, data: pd.DataFrame, group_col: str, positive_rate_col: str, total_col: str, alpha=0.05):
        """
        Initialize the AB Testing Tool for proportions.
        
        Parameters:
        - data (pd.DataFrame): The dataset containing group labels, positive rates, and total sample sizes.
        - group_col (str): The column name for group labels.
        - positive_rate_col (str): The column name for positive rates (percentages).
        - total_col (str): The column name for total sample sizes.
        - alpha (float): The significance level for testing.
        """
        self.data = data
        self.group_col = group_col
        self.positive_rate_col = positive_rate_col
        self.total_col = total_col
        self.alpha = alpha
        self.results = None

    def calculate_counts(self):
        """
        Calculate the count of positive outcomes for each group.
        Adds a new column 'positive_count' to the data.
        """
        self.data["positive_count"] = (self.data[self.positive_rate_col] / 100) * self.data[self.total_col]
        self.data["positive_count"] = self.data["positive_count"].round().astype(int)

    def test_two_groups(self):
        """
        Perform a Z-Test for proportions for two groups.
        Returns:
        - str: The name of the test used.
        - dict: The result of the statistical test.
        """
        groups = self.data[self.group_col].unique()
        if len(groups) != 2:
            raise ValueError("This method is only for two groups. Use `test_multiple_groups` for more than two groups.")
        
        group_data = self.data.groupby(self.group_col).sum()
        count = group_data.loc[groups, "positive_count"].values
        nobs = group_data.loc[groups, self.total_col].values

        stat, p = proportions_ztest(count=count, nobs=nobs)

        if stat > 0:
            significance = f"Group '{groups[0]}' is significantly greater than '{groups[1]}'"
        else:
            significance = f"Group '{groups[1]}' is significantly greater than '{groups[0]}'"

        if p >= self.alpha:
            significance = "No significant difference between groups"

        significance = "significant" if p < self.alpha else "not significant"
        return "Proportions Z-Test", {"statistic": stat, "p_value": p, "significance": significance}

    def test_multiple_groups(self):
        """
        Perform a Chi-Square Test for multiple groups.
        Returns:
        - str: The name of the test used.
        - dict: The results of the statistical test, including pairwise comparisons.
        """
        groups = self.data[self.group_col].unique()
        
        # Create contingency table
        contingency_table = self.data.pivot_table(index=self.group_col,
                                                  values=["positive_count", self.total_col],
                                                  aggfunc="sum")
        contingency_table["negative_count"] = contingency_table[self.total_col] - contingency_table["positive_count"]
        observed = contingency_table[["positive_count", "negative_count"]].values
        
        # Chi-Square Test
        chi2, p, _, _ = chi2_contingency(observed)
        significance = "significant" if p < self.alpha else "not significant"

        results = {"statistic": chi2, "p_value": p, "significance": significance}
        
        if p < self.alpha:
            # Perform post-hoc pairwise comparisons
            pairwise_results = self.post_hoc_test()
            results["pairwise"] = pairwise_results
        
        return "Chi-Square Test", results

    def post_hoc_test(self):
        """
        Perform post-hoc pairwise Z-Tests for multiple groups.
        Returns:
        - pd.DataFrame: Pairwise comparison results including significance.
        """
        pairs = list(combinations(self.data[self.group_col].unique(), 2))
        results = []

        for group1, group2 in pairs:
            data1 = self.data[self.data[self.group_col] == group1]
            data2 = self.data[self.data[self.group_col] == group2]
            
            count = [
                data1["positive_count"].values[0],
                data2["positive_count"].values[0]
            ]
            nobs = [
                data1[self.total_col].values[0],
                data2[self.total_col].values[0]
            ]
            
            stat, p = proportions_ztest(count=count, nobs=nobs)
            significance = "significant" if p < self.alpha else "not significant"
            
            results.append({
                "group1": group1,
                "group2": group2,
                "statistic": stat,
                "p_value": p,
                "significance": significance
            })
        
        return pd.DataFrame(results)

    def run(self):
        """
        Run the appropriate statistical test based on the number of groups.
        Returns:
        - dict: Results of the test.
        """
        self.calculate_counts()
        groups = self.data[self.group_col].unique()
        
        if len(groups) == 2:
            test_used, results = self.test_two_groups()
        else:
            test_used, results = self.test_multiple_groups()
        
        self.results = {"test_used": test_used, "results": results}
        return self.results


if __name__ == "__main__":
    print("Welcome to the AB Testing Tool!")
    
    # Gather inputs interactively
    num_groups = int(input("Enter the number of groups: "))
    
    groups = []
    positive_rates = []
    total_sends = []
    
    for i in range(num_groups):
        group = input(f"Enter the name of group {i + 1}: ")
        positive_rate = float(input(f"Enter the positive rate (percentage) for group {group}: "))
        total_send = int(input(f"Enter the total number of sends for group {group}: "))
        
        groups.append(group)
        positive_rates.append(positive_rate)
        total_sends.append(total_send)
    
    alpha = float(input("Enter the significance level (default is 0.05): ") or 0.05)
    
    # Create the DataFrame
    data = pd.DataFrame({
        "group": groups,
        "positive_rate": positive_rates,
        "total_sends": total_sends
    })
    
    # Run the tool
    tool = ABTestProportionsTool(data=data, group_col="group", positive_rate_col="positive_rate", total_col="total_sends", alpha=alpha)
    results = tool.run()
    
    print("\nResults:")
    print(results)
