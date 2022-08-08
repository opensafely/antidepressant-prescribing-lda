# study start date.  should match date in project.yaml
start_date = "2020-01-01"

# study end date.  should match date in project.yaml
end_date = "2020-04-01"

# depression register start date
depr_register_date = "2006-04-01"
# day to check for ongoing depression
register_ongoing_date = "2006-03-31"

# demographic variables by which code use is broken down
demographics = [
    "age_band",
    "sex",
    "region",
    "carehome",
    "learning_disability",
    "imd",
    "ethnicity",
]

# Subgroups for LDA population plots
# We extract the categorical variables so that we can programmatically use
# them as labels. We still need to extract the numerical group for use in
# the measures framework
# Key is the label, value is the numerical
lda_subgroups = {"autism": "aut", "learning_disability": "ld"}

# name of measure
marker = "Depression review"
