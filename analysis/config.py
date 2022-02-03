#study start date.  should match date in project.yaml
start_date = "2019-01-01"

#study end date.  should match date in project.yaml
end_date = "2022-01-01"

#demographic variables by which code use is broken down
demographics = ["sex", "imd", "ethnicity"]

#name of measure
marker="Depression review"

#TODO: why do we separately need to specify this here? Can we do this cleaner?
#codelist path
codelist_path = "codelists/nhsd-primary-care-domain-refsets-deprvw_cod.csv"
