import pandas

if __name__ == "__main__":
    maoi = pandas.read_csv("maoi_op.csv", parse_dates=["date"])
    tri = pandas.read_csv("tricyclic_op.csv", parse_dates=["date"])
    ssri = pandas.read_csv("ssri_op.csv", parse_dates=["date"])
    other = pandas.read_csv("other_op.csv", parse_dates=["date"])

    maoi["type"] = "maoi"
    tri["type"] = "tri"
    ssri["type"] = "ssri"
    other["type"] = "other"

    df = pandas.concat([maoi, tri, ssri, other])
    region_sum = df.groupby(["date", "name"]).agg(
        {"y_items": sum, "total_list_size": lambda x: x.iloc[0]}
    )
    region = region_sum.reset_index()
    total_sum = region.groupby(["date"]).sum()
    total = total_sum.rename(
        {"y_items": "numerator", "total_list_size": "denominator"}, axis=1
    )
    total["value"] = total.numerator / total.denominator
    total["name"] = "antidepressant_any_all_openprescribing_total_rate"
    total["group_0"] = "population"
    total["category_0"] = "population"
    total["group_1"] = None
    total["category_1"] = None
