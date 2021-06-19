from typing import Collection, Dict, List, Union

import funcy
import ff
import pandas as pd
from tqdm import tqdm


@funcy.memoize
def _select(var):
    return ff.select(var)


def clean_garbage_chars(s: Optional[str]) -> Optional[str]:
    return ''.join(c for c in s if c.isascii()) if s else s


def info(variables: Union[str, Collection[str]]) -> pd.DataFrame:
    """Get metadata on each variable

    An explanation of the resulting metadata can be found here: http://metadata.fragilefamilies.princeton.edu/about
    """  # noqa
    if isinstance(variables, str):
        variables = [variables]
    records = []
    for var in tqdm(variables):
        records.append(_select(var))
    result = pd.DataFrame.from_records(records)
    for col in ['probe', 'qtext']:
        result[col] = result[col].apply(clean_garbage_chars)
    return result


def search(query: Union[Dict, List[Dict]]) -> List:
    """Search for variables that appear in FFC

    The query API is identical to https://github.com/fragilefamilieschallenge/ffmetadata-py. The addition here is that we add a filter to ever query to require that the variable appears in FFC. So users should not pass this filter themselves.

    Examples::

       # search for 'school' appearing in the label (i.e. description)
       search({'name': 'label', 'op': 'like', 'val': '%school%})

       # search for all questions asked to the Father in the Baseline wave
       search([
           {'name': 'survey', 'op': 'eq', 'val': 'Father'},
           {'name': 'wave', 'op': 'eq', 'val': 'Baseline'},
       ])
    """
    if isinstance(query, dict):
        query = [query]
    query = [
        {'name': 'in_FFC_file', 'op': 'eq', 'val': 'Yes'},
        *query,
    ]
    return ff.search(query)


searchinfo = funcy.compose(info, search)
"""Search and then get info on search results"""
