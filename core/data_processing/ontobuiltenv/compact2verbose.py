from core.sparql import SparqlQuery
from core.sparql.graph_pattern import GraphPattern, OptionalClause, TriplePattern
from core.sparql.query_form import SelectClause


class OBECompact2VerboseConverter:
    def _try_convert_property_hasmeasure_triple(self, pattern: GraphPattern):
        try:
            """
            ?Property obe:has{key} ?{key} .
            """
            assert isinstance(pattern, TriplePattern)
            assert pattern.subj == "?Property"
            assert len(pattern.tails) == 1

            p, o = pattern.tails[0]

            key = p[len("obe:has"):]
            assert key in ["TotalFloorArea", "MarketValue", "GroundElevation", "TotalMonetaryValue", "MonetaryValue"]
            assert o == "?" + key

            """
            ?Property obe:has{key} ?{key} .
            ?{key} om:hasValue [
                om:hasNumericalValue ?{key}NumericalValue ; 
                om:hasUnit/om:symbol ?{key}Unit
            ] .
            """
            patterns = [
                pattern,
                TriplePattern(
                    "?{key}".format(key=key),
                    tails=[
                        (
                            "om:hasValue",
                            "[ om:hasNumericalValue ?{key}NumericalValue ; om:hasUnit/om:symbol ?{key}Unit ]".format(key=key),
                        )
                    ],
                )
            ]
            vars = ["?{key}NumericalValue".format(key=key), "?{key}Unit".format(key=key)]
            return patterns, vars
        except AssertionError:
            return None

    def _try_convert_property_hasaddress_triple(self, pattern: GraphPattern):
        try:
            """
            ?Property obe:hasAddress ?Address .
            """
            assert isinstance(pattern, TriplePattern)
            assert pattern.subj == "?Property"
            assert len(pattern.tails) == 1

            p, o = pattern.tails[0]
            assert p == "obe:hasAddress"
            assert o == "?Address"

            """
            ?Property obe:hasAddress ?Address .
            OPTIONAL {
                ?Address ict:hasBuilidingName ?BuildingName .
            }
            OPTIONAL {
                ?Address ict:hasStreet ?Street .
            }
            OPTIONAL {
                ?Address ict:hasStreetNumber ?StreetNumber .
            }
            OPTIONAL {
                ?Address obe:hasUnitName ?UnitName .
            }
            OPTIONAL {
                ?Address obe:hasPostalCode/rdfs:label ?PostalCode
            }
            """

            patterns = [
                pattern,
                OptionalClause([TriplePattern.from_triple("?Address", "ict:hasBuildingName", "?Building")]),
                OptionalClause([TriplePattern.from_triple("?Address", "ict:hasStreet", "?Street")]),
                OptionalClause([TriplePattern.from_triple("?Address", "ict:hasStreetNumber", "?StreetNumber")]),
                OptionalClause([TriplePattern.from_triple("?Address", "obe:hasUnitName", "?UnitName")]),
                OptionalClause([TriplePattern.from_triple("?Address", "obe:hasPostalCode/rdfs:label", "?PostalCode")])
            ]
            vars = ["?BuildingName", "?Street", "?StreetNumber", "?UnitName", "?PostalCode"]

            return patterns, vars
        except AssertionError:
            return None

    def convert(self, sparql_compact: SparqlQuery):
        select_vars_verbose = list(sparql_compact.select_clause.vars)
        graph_patterns_verbose = []

        for pattern in sparql_compact.graph_patterns:
            optional = self._try_convert_property_hasmeasure_triple(pattern)
            if optional is not None:
                patterns, select_vars = optional
                select_vars_verbose.extend(select_vars)
                graph_patterns_verbose.extend(patterns)
                continue

            optional = self._try_convert_property_hasaddress_triple(pattern)
            if optional is not None:
                patterns, select_vars = optional
                select_vars_verbose.extend(select_vars)
                graph_patterns_verbose.extend(patterns)
                continue

            graph_patterns_verbose.append(pattern)

        return SparqlQuery(
            select_clause=SelectClause(
                solution_modifier="DISTINCT", vars=select_vars_verbose
            ),
            graph_patterns=graph_patterns_verbose,
        )
