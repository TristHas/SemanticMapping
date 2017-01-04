#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Linking to WN 2.0

###
###     OLD
###


# Linking to WN 3.1
def link_ImageNet_to_wn31():
    with open(link_file, "r") as f:
        tab = [[],[]]
        for line in f:
            tmp = line.split()
            tab[0].append("n" + tmp[1])
            tab[1].append(tmp[2])
    mapping = pd.Series(index = tab[0], data = tab[1], name = "wn31synsetID")
    return in_data.join(mapping)



def link_ImageNet_to_wn20():
    """
        SHOULD LAUNCH FUSEKI WITH CORRECT SERVICE HERE
    """
    def ImagenetID_to_WN30SynsetID(ImageNetID):
        return "1" + ImageNetID[1:]

    def map_wn20synset_wn30SynsetID():
        statement = """
        SELECT DISTINCT ?wn30synsId ?wn20syns ?g
            WHERE {
                GRAPH <http://mydataset/wordnet-synset.ttl> {
                    ?wn30syns  <http://www.w3.org/2006/03/wn/wn20/schema/synsetId> ?wn30synsId .
                    ?wn30syns a <http://www.w3.org/2006/03/wn/wn20/schema/NounSynset>
                }
                GRAPH ?g {
                    ?wn30syns  <http://purl.org/dc/terms/replaces> ?wn20syns .
                }
            }
        """
        result = np.asarray([(y[0].value, y[1].value, y[2].value) for y in cl.req(statement)])
        return pd.DataFrame(data = {"wn20synset"        :result[:,1],
                                    "mapping_confidence":result[:,2],
                                   },
                            index = result[:,0])

    endpoint = "http://localhost:3030/ds/query"
    cl                  = Client(endpoint)

    # Get Imagenet Metadata
    in_data                     = load_syns_metadata()
    # Set index to wn30synsID
    in_data["ImagenetID"]       = in_data.index.to_series()
    in_data                     = in_data.set_index(in_data["ImagenetID"].apply(ImagenetID_to_WN30SynsetID))

    # Add Mapping as "wn20synset" and "mapping_confidence" columns.
    # in_data_mapped contains all Imagenet wnid
    # in_im_and_map contains Imagenet wnid having both images and wn20 mapped synset(s)
    map                         = map_wn20synset_wn30SynsetID()
    in_data_mapped              = in_data.join(map)
    in_im_and_map               = in_data_mapped[in_data_mapped.numImage.notnull() & in_data_mapped.wn20synset.notnull()]

    return in_data_mapped, in_im_and_map


def map_wn20synset_dbpres():
    endpoint = "http://localhost:3030/ds/query"
    cl                  = Client(endpoint)

    statement = """
    SELECT DISTINCT ?wn20syns ?dbpres
        WHERE {
            GRAPH ?g {
                ?dbpres <http://dbpedia.org/property/wordnet_type> ?wn20syns.
            }
        }
    """
    result = np.asarray([(y[0].value, y[1].value) for y in cl.req(statement)])
    return pd.DataFrame(data = {"dbpres"        :result[:,1]
                               },
                        index = result[:,0])


def what():
    wn_bn_dbp_file = "/home/tristan/data/wnid_bnid_dbpres"
    wn_bn_file      ="/home/tristan/data/wnid_bnid"
    in_data = load_in_wn31_mapped()
    wn_bn_data = pd.read_pickle(wn_bn_file)
    wn_bn_dbp_data = pd.read_pickle(wn_bn_dbp_file)

