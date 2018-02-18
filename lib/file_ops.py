def load_list_from_file(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines


def load_dict_from_file(filename):
    d = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            d[parts[0]] = parts[1]
    return d


def save_list_to_file(lines, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            print(line, file=file)


def save_dict_to_file(dict, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key in dict:
            print('{} {}'.format(key, dict[key]), file=file)
