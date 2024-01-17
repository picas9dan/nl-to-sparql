from dataclasses import dataclass
from typing import Optional
from core.sparql.query_form import SelectClause
from core.sparql.solution_modifier import SolutionModifier
from core.sparql.sparql_base import SparqlBase
from core.sparql.where_clause import WhereClause


@dataclass(order=True, frozen=True)
class SparqlQuery(SparqlBase):
    select_clause: SelectClause
    where_clause: WhereClause
    solultion_modifier: Optional[SolutionModifier] = None

    def __str__(self):
        text = "{select_clause} {where_clause}".format(
            select_clause=self.select_clause,
            where_clause=self.where_clause,
        )
        if self.solultion_modifier:
            text += "\n" + str(self.solultion_modifier)
        return text

    @classmethod
    def fromstring(cls, sparql: str):
        select_clause, sparql_fragment = SelectClause.extract(sparql)
        where_clause, sparql_fragment = WhereClause.extract(sparql_fragment)
        solution_modifier, sparql_fragment = SolutionModifier.extract(sparql_fragment)
        assert not sparql_fragment or sparql_fragment.isspace(), sparql_fragment
        return cls(select_clause, where_clause, solution_modifier)
