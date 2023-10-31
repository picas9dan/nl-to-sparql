from core.data_processing.correct_predicate import PredicateCorrector
from core.data_processing.correct_span import SpanCorrector
from core.data_processing.sparql import SparqlQuery, TriplePattern, ValuesClause


class SparqlPredictionCorrector:
    def __init__(self):
        self.predicate_corrector = PredicateCorrector()
        self.span_corrector = SpanCorrector()

    def correct_values_clause(self, values_clause: ValuesClause, nlq: str):
        values = []
        for value in values_clause.values:
            value = self.span_corrector.correct(nlq, value)
            values.append(value)
        return ValuesClause(values_clause.var, values)
    
    def correct_triple_pattern(self, triple_pattern: TriplePattern):
        tails = []
        for predicate, obj in triple_pattern.tails:
            predicate = self.predicate_corrector.correct(predicate)
            tails.append((predicate, obj))
        return TriplePattern(subj=triple_pattern.subj, tails=tails)

    def correct(self, sparql: SparqlQuery, nlq: str):
        graph_patterns = []
        for pattern in sparql.graph_patterns:
            if isinstance(pattern, ValuesClause):
                pattern = self.correct_values_clause(pattern, nlq)
            elif isinstance(pattern, TriplePattern):
                pattern = self.correct_triple_pattern(pattern)
            graph_patterns.append(pattern)

        return SparqlQuery(sparql.select_clause, graph_patterns)