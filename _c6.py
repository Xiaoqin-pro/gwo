import io
p = r"D:\111\Desktop\噜噜\狼群\src\algorithmConfig.m"
s = io.open(p, encoding="utf-8").read()
s = s.replace("    case 'IGWO',",
              "    case 'IGWO',         cfg=struct('f',@IGWO,'N',50,'state',struct());\n"
              "    case 'RASE-IGWO',    cfg=struct('f',@IGWO,'N',50,'state',struct('evalPolicy','screen', ...\n"
              "                             'K',50,'screenPolicy','geometry','divCriterion','minority','convShare',0.9));\n"
              "    case 'RASE-IGWO-r',  cfg=struct('f',@IGWO,'N',50,'state',struct('evalPolicy','screen', ...\n"
              "                             'K',50,'screenPolicy','random','convShare',0.9));\n"
              "    case 'RASE-IGWO-K25',cfg=struct('f',@IGWO,'N',50,'state',struct('evalPolicy','screen', ...\n"
              "                             'K',25,'screenPolicy','geometry','divCriterion','minority','convShare',0.9));\n"
              "    case 'RASE-IGWO-full',cfg=struct('f',@IGWO,'N',50,'state',struct('evalPolicy','full'));", 1)
s = s.replace("    case 'IGWO',         cfg=struct('f',@IGWO,'N',50,'state',struct());\n    case 'IGWO',", "    case 'IGWO',")
io.open(p, "w", encoding="utf-8").write(s)
print("configs added")
