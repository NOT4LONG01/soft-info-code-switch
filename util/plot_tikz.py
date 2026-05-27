### Write data to tikz data file for plotting in latex
dimPath = "../latex_notes/"
dimBoundName = 'data/dimBounds_l=%d_%s.dat'%(ell, codeType)
dimBoundFile = dimPath + dimBoundName

fBound =  open(dimBoundFile, 'w')
fBound.write("r dim_ub dim_lb \n") # Write column labels

r_max = 2**(ell-2)+1
for r in range(1, r_max):
    ub = dim_ub(ell,r)
    lb = dim_lb(ell,r)
    fBound.write("%d %f %f \n"%(r, ub, lb)) # Write data to corresponding column

fBound.close()

### Generate tikzpicture for latex
figPath = "../latex_notes/"
figFileName = "tikz/dimPlot_l=%d_log.tex"%(ell) # parameterized file name
figFile = figPath + figFileName

xmax = min([q,ceil(q/4)+5]) # max value of x axis in plot
ymax = min([q^ceil(log(dim_ub(ell,1),q))/q^2, 1]) # max value of y axis in plot 

with open(figFile,'w') as fFig:

    fFig.write("\
    \\begin{tikzpicture}\n\
    \\pgfplotsset{compat = 1.3}\n\
    \\begin{axis}[\n\
    legend style={nodes={scale=0.5, transform shape}},\n\
    cycle list name = {sims_list},\n\
    width = 0.9\\columnwidth,\n\
    height = 0.6\\columnwidth,\n\
    xlabel = {{Local Redundancy $r$}},\n\
    ylabel = {{Rate of Curve-lifted RS codes}},\n\
    ymode=log,\n\
    log basis y={%d}, \n\
    xmin = 1,\n\
    xmax = %d,\n\
    ymin = 0,\n\
    ymax = %d,\n\
    legend pos = south west,\n\
    legend cell align=left,\n\
    grid=both]\n\n"%(q, xmax, ymax))

    fFig.write("\n\\addplot table[x=r, y=rate_ub] {"+dimBoundName+"};\n")
    fFig.write("\n\\addlegendentry{{CL dim. (ub) $\\ell=%d$}};\n"%ell)

    fFig.write("\n\\addplot table[x=r, y=rate_lb] {"+dimBoundName+"};\n")
    fFig.write("\n\\addlegendentry{{CL dim. (lb) $\\ell=%d$}};\n"%ell)

    fFig.write("\
    \\end{axis}\n\
    \\end{tikzpicture}")

### Add an entry in loadPlots.tex to show the plots
loadPath = "../latex_notes/loadPlots.tex"
n = (2^ell)^2
with open(loadPath, 'a') as fLoad:
    fLoad.write("\
    \\begin{figure}[h]\n\
    \\centering\n\
    \\input{%s}\n\
    \\caption{rate log plot $\\ell=%d, n=q^2=%d$.}\n"%(figFileName, ell, n))
    fLoad.write("\\end{figure}\n\n")
