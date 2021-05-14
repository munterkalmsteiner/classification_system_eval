import os.path
from au import AnalysisUnit
import treeify
import utils
import metrics
from gensim.models.doc2vec import Doc2Vec

results_path = "results"

print('Loading doc2vec models...')
model_en = Doc2Vec.load("models/wikipedia_en_20210308")

csystems = [
    { "name": "uniclass", "creator": treeify.uniclass, "d2vmodel": model_en},
    { "name": "omniclass","creator": treeify.omniclass, "d2vmodel": model_en},
    { "name": "naics", "creator": treeify.naics, "d2vmodel": model_en},
    { "name": "nace", "creator": treeify.nace, "d2vmodel": model_en},
    { "name": "eucyber", "creator": treeify.eucyber, "d2vmodel": model_en},
    { "name": "mahaini", "creator": treeify.mahaini, "d2vmodel": model_en}
]


for csystem in csystems:
    csfile = f'{results_path}/{csystem["name"]}.tree'
    csname = csystem["name"]
    if os.path.isfile(csfile):
        print(f'Using existing {csname} tree stored in: {csfile}')
        cstree = utils.load_object(csfile)
    else:
        print(f'Creating new {csname} tree...')
        cstree = csystem["creator"](csname)
        utils.save_object(cstree, csfile)
    csystem["tree"] = cstree

    print(f'Calculating conciseness for {csname}...')
    csystem["conciseness"] = metrics.conciseness(cstree, 0)
    utils.save_text(csystem["conciseness"], f'{results_path}/{csystem["name"]}.cc')

    aufile = f'{results_path}/{csystem["name"]}.au'
    if os.path.isfile(aufile):
        print(f'Using existing {csname} analysis units file: {aufile}')
        aunits = utils.load_object(aufile)
    else:
        if csystem["d2vmodel"] is not None:
            print(f'Creating new {csname} analysis units...')
            aunits = metrics.create_analysis_units(cstree, csystem["d2vmodel"])
            utils.save_object(aunits, f'{results_path}/{csystem["name"]}.au')
            utils.save_analysis_units_description(aunits, f'{results_path}/{csystem["name"]}.au.txt')

    if csystem["d2vmodel"] is not None:
        print(f'Calculating robustness for {csname}...')
        csystem["robustness"] = metrics.robustness(aunits)
        utils.save_text(csystem["robustness"], f'{results_path}/{csystem["name"]}.rb')
