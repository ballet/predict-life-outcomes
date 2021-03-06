import logging
from typing import Collection, Dict, List, Optional, Union

import ff
import pandas as pd
from funcy import first, memoize
from tqdm import tqdm

logger = logging.getLogger(__name__)


@memoize
def _select(var: str, retry_with_old_name=True) -> dict:
    try:
        return ff.select(var)
    except Exception as e:
        if 'Invalid variable name' in str(e):
            if retry_with_old_name:
                new_name = _find_new_name(var)
                if new_name:
                    return _select(new_name, retry_with_old_name=False)
        raise ValueError(f'Error in getting metadata for {var!r}') from e


def _find_new_name(var: str) -> Optional[str]:
    matches = search({'name': 'old_name', 'op': 'eq', 'val': var})
    return first(matches)


def clean_garbage_chars(s: Optional[str]) -> Optional[str]:
    return ''.join(c for c in s if c.isascii()) if s else s


def info(
    variables: Union[str, Collection[str]],
    retry_with_old_name=True
) -> pd.DataFrame:
    """Get metadata on each variable

    An explanation of the resulting metadata can be found here: http://metadata.fragilefamilies.princeton.edu/about
    """  # noqa
    if isinstance(variables, str):
        variables = [variables]
    records = []
    for var in tqdm(variables):
        records.append(_select(var, retry_with_old_name=retry_with_old_name))
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        for col in ['probe', 'qtext']:
            result[col] = result[col].apply(clean_garbage_chars)
    return result


def search(query: Union[Dict, List[Dict]]) -> List[str]:
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
    """  # noqa
    if isinstance(query, dict):
        query = [query]
    query = [
        {'name': 'in_FFC_file', 'op': 'eq', 'val': 'Yes'},
        *query,
    ]
    return ff.search(query)


def searchinfo(
    query: Union[Dict, List[Dict]],
    limit: Optional[int] = 30,
) -> pd.DataFrame:
    """Search and then get info on search results

    See metadata.search for details on search queries.

    Args:
        query: query for metadata.search
        limit: max number of items to return, set to None to see info for all
            items (defaults to 30)
    """
    variables = search(query)
    if limit and limit < len(variables):
        logger.info(
            f'Returning info for {limit} variables (out of '
            f'{len(variables)} found); pass limit=None to see info for all '
            'variables.')
    return info(variables[:limit])
