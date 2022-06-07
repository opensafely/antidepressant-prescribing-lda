"""Comparison functions for Series that may include NaN.

The PRIMIS spec includes logic for assigning patients to groups based on whether the
latest code in one codelist appears before/after the latest code from another codelist.

For instance, patients are assigned to DIAB_GROUP on the condition DIAB_DAT > DMRES_DAT.

The following does not work:

    df["DIAB_GROUP"] = (df["DIAB_DAT"] > df["DMRES_DAT"])

because:

    >>> np.nan == np.nan
    False
    >>> np.nan > np.nan
    False

etc etc.

In this example, the patient should be assigned to DIAB_GROUP if either:

    * neither DIAB_DAT and DMRES_DAT are null, and DIAB_DAT > DMRES_DAT
    * DIAB_DAT is not null and DMRES_DAT is null

And the patient should not be assigned to DIAB_GROUP if one of:

    * neither DIAB_DAT and DMRES_DAT are null, and DIAB_DAT <= DMRES_DAT
    * DIAB_DAT is null and DMRES_DAT is not null
    * DIAB_DAT and DMRES_DAT are both null

As such, we need the functions below, which handle NaN values correctly.
"""
import logging


def gt(lhs, rhs):
    return (lhs > rhs) | (lhs.notna() & rhs.isna())


def gte(lhs, rhs):
    logging.error(lhs, rhs)
    return (lhs >= rhs) | (lhs.notna() & rhs.isna())


def lt(lhs, rhs):
    return (lhs < rhs) | (lhs.isna() & rhs.notna())


def lte(lhs, rhs):
    return (lhs <= rhs) | (lhs.isna() & rhs.notna())
