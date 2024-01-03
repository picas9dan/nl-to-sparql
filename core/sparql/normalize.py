from typing import List

from core.sparql import SparqlQuery
from core.sparql.graph_pattern import BindClause, FilterClause, GraphPattern, OptionalClause, TriplePattern, ValuesClause


def make_canonical(query: SparqlQuery):
    """
    Ordering:
      - VALUES
      - Triple
      - OPTIONAL
      - BIND
      - FILTER
    """
    values_clauses: List[ValuesClause] = []
    triples: List[TriplePattern] = []
    optional_clauses: List[OptionalClause] = []
    bind_clauses: List[BindClause] = []
    filter_clauses: List[FilterClause] = []

    for pattern in query.graph_patterns:
        if isinstance(pattern, ValuesClause):
            values_clauses.append(pattern)
        elif isinstance(pattern, TriplePattern):
            triples.append(pattern)
        elif isinstance(pattern, OptionalClause):
            optional_clauses.append(pattern)
        elif isinstance(pattern, BindClause):
            bind_clauses.append(pattern)
        elif isinstance(pattern, FilterClause):
            filter_clauses.append(pattern)
        else:
            raise ValueError("Unrecognized pattern type: " + pattern)
        
    values_clauses.sort()
    triples.sort()
    optional_clauses.sort()
    bind_clauses.sort()
    filter_clauses.sort()

    patterns: List[GraphPattern] = []
    for lst in (values_clauses, triples, optional_clauses, bind_clauses, filter_clauses):
        patterns.extend(lst)

    return SparqlQuery(query.select_clause, patterns)


def normalize_query(query: str):
    try:
        return str(make_canonical(SparqlQuery.fromstring(query)))
    except Exception:
        return query
