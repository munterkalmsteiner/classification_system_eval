import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inputf:
        return pickle.load(inputf)

def save_text(txt, filename):
    with open(filename, 'w') as output:
        output.write(txt)

def save_analysis_units_description(units, filename):
    ret = ""
    for unit in sorted(units, reverse = True):
        ret = ret + unit.describe()

    save_text(ret, filename)
