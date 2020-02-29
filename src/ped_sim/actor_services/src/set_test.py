from lxml import etree

f = "/home/lintao/a.world"

tree = etree.parse(f)
world = tree.getroot().getchildren()[0]

s = [[-22, 0], [0, -2]]
g = [[222, 0], [0, 22]]

for element in world.iter('actor'):
    index = int(element.get('name')[-1])
    # print ("{}, {}".format(element.tag, index))
    for p in element.iter('pose'):
        # p.text = str(s[index][0]) + ' ' + str(s[index][1]) + ' 1.02 0 0'
        x = str(s[index][0])
        y = str(s[index][1])
        pose = [x, y, '1.02', '0', '0']
        p.text = ' '.join(pose)

    for t in element.iter('target'):
        # print ("{}, {}".format(t.tag, t.text))
        x = str(g[index][0])
        y = str(g[index][1])
        position = [x, y, '1.02']
        t.text = ' '.join(position)

tree.write(f, xml_declaration=True, encoding="utf-8", pretty_print=True)
