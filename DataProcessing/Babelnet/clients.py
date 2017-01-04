#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sparql

class Client(object):
    def __init__(self, endpoint):
        self.s = sparql.Service(endpoint)

    def req(self, statement):
        result = self.s.query(statement)
        return result.fetchall()

class Client_tests(object):
    def __init__(self, endpoint, prefixes):
        self.s = sparql.Service(endpoint)
        self.prefix =  " \n ".join(["PREFIX {}: {}".format(key, prefixes[key]) for key in prefixes])
        self.e = {'prefix':self.prefix}

    def p_s(self, p_namespace, p):
        statement = """ {prefix}
                        SELECT DISTINCT ?x WHERE {{
                                {}:{} ?x ?z.
                        }}
                        limit 500
                        """.format(p_namespace, p, **self.e)
        result = self.s.query(statement)
        return result.fetchall()

    def po_s(self, p_namespace, p):
        statement = """ {prefix}
                        SELECT DISTINCT ?x ?z WHERE {{
                                {}:{} ?x ?z.
                        }}
                        limit 500
                        """.format(p_namespace, p, **self.e)
        print statement
        result = self.s.query(statement)
        return result.fetchall()


    def o_sp(self, res, prop):
        statement = """ {prefix}
                        SELECT ?z WHERE {{
                                dbres:{} dbprop:{} ?z.
                        }}
                        limit 500
                        """.format(res, prop, **self.e)

        result = self.s.query(statement)
        return result.fetchall()

    def s_p(self, prop):
        statement = """ {prefix}
                        SELECT DISTINCT ?x WHERE {{
                                ?x dbprop:{} ?z.
                        }}
                        limit 500
                        """.format(prop, **self.e)
        result = self.s.query(statement)
        return result.fetchall()


    def type(self, res):
        statement = """ {prefix}
                        SELECT DISTINCT ?z WHERE {{
                                dbres:{} w3:22-rdf-syntax-ns#type ?z.
                        }}
                        limit 500
                        """.format(res, **self.e)
        result = self.s.query(statement)
        return result.fetchall()

    def named_graphs(self):
        statement = """SELECT DISTINCT ?g
                    WHERE {
                      GRAPH ?g {
                        ?s ?p ?o
                      }
                    }
                    """
        result = self.s.query(statement)
        return result.fetchall()

    def graph_prop(self, graph, limit = False):
        if not limit:
            limit = self.limit
        statement = """
                    {prefix}
                    SELECT DISTINCT ?p
                    WHERE {{
                      GRAPH :{} {{
                        ?s ?p ?o
                      }}
                    }} limit {}
                    """.format(graph, limit, **self.e)
        result = self.s.query(statement)
        return result.fetchall()

    def graph_sample(self, graph, limit = False):
        if not limit:
            limit = self.limit
        statement = """
                    {prefix}
                    SELECT DISTINCT ?s ?p ?o
                    WHERE {{
                      GRAPH :{} {{
                        ?s ?p ?o
                      }}
                    }} limit {}
                    """.format(graph, limit, **self.e)
        result = self.s.query(statement)
        return result.fetchall()



