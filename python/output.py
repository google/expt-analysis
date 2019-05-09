#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# author: Reza Hosseini

## create a directory if it does not exist
def CreateDir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

#### this function creates a latex file skin. i.e. the main document
def LatexSkin(fn, docType='article', figsPath='{figs/}',
              latexInputFn='Input.tex'):
  latex1 = '\\documentclass{' + docType + '}'
  latex2 = ('\n \\usepackage{enumerate,amsmath,graphics,amssymb,graphicx,' +
            'amscd,amscd,amsbsy,multirow,float,booktabs,verbatim,xy,' +
            'geometry,import} \n')
  latex3 = '\n \\graphicspath{' + figsPath + '} \n'
  latex4 = ('\n \\begin{document} \n \\input{' + latexInputFn +
            '} \n \\end{document}')

  with open(fn, 'w') as f:
    f.write(latex1)
    f.write(latex2)
    f.write(latex3)
    f.write(latex4)

#### this function writes the text in the center for a given latex file
def WriteLatex(fn, text):
    with open(fn, 'a') as f:
        f.write('\n' + text + '\n')

### this begins a frame in latex
def BeginFrame(fn, frameTitle=''):
    with open(fn, 'a') as f:
        f.write('\n' + '\\begin{frame}' + '\n')
        f.write('\n' + '\\frametitle{' + frameTitle +'}' + '\n')

### this ends a frame in latex
def EndFrame(fn):
    with open(fn, 'a') as f:
        f.write('\n' + '\\end{frame}' + '\n')

### this function writes the text in the center for a given latex file
def WriteLatexCenter(fn, text):
    with open(fn, 'a') as f:
        f.write('\\begin{center} \n ')
        f.write(text + '\n')
        f.write('\\end{center} \n ')

### this function adds a figure with caption and label to a latex file
def LatexFig(latexFn, figFn, figLabel=None, figCaption=None, scale=str(0.5)):
  with open(latexFn, 'a') as f:
    f.write('\n \\begin{figure}[H] \n')
    f.write('\\centering \n')
    f.write('\\includegraphics[scale=' + scale + ']{' + figFn + '} \n')
    if figCaption is not None:
      f.write('\\caption{' + figCaption + '} \n')
    if figLabel is not None:
      f.write('\\label{' + figLabel + '} \n')
    f.write('\\end{figure} \n')


############################ Creating figures ################
def Pdff(fn, PlotFcn, plotTitle):

  with PdfPages(fn) as pdf:
    plt.figure(figsize=(3, 3))
    PlotFcn()
    plt.title(pltTitle)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

def SaveFig(fn, PlotFcn, plotTitle='', format='png', dpi=100):
  PlotFcn()
  import matplotlib
  if plotTitle != '':
    matplotlib.pylab.title(plotTitle)
  matplotlib.pylab.savefig(fn, format=format, dpi=dpi)


'''def PlotFcn(): pylab.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
PlotFcn()
figFn = ofigs+'fig1.pdf'
plotTitle = 'maast & kashk'
pdff(figFn,PlotFcn,plotTitle)
fn = odir + 'skin.tex'
LatexSkin(fn,docType='article')
fn = odir + 'Input.tex'
with open(fn, 'a') as f: f.write('This is an automatically generated EDA')
fn = odir + 'Input.tex'
LatexFig(fn,'fig1.pdf','figLabel','mofo')'''


### save fig and add to input latex file
def SaveFig_addLatex(figCaption, latexFn, ofigs, figFn='', format='png',
                     dpi=100, skipCaption=False, skipLabel=False):
  import matplotlib

  if figFn == '':
    figFn=figCaption.replace(' ','_')

  figFn = ofigs + figFn + '.png'
  matplotlib.pylab.savefig(figFn, format=format, dpi=dpi)
  LatexFig(latexFn=latexFn, figFn=figFn, figLabel=figFn,
           figCaption=figCaption, skipCaption=skipCaption,
           skipLabel=skipLabel)

def SaveFig_addSlide(figCaption, latexFn, ofigs, figFn='', format='png',
                     dpi=100, skipCaption=False, skipLabel=False):

  import matplotlib
  if figFn == '':
    figFn=figCaption.replace(' ','_')
  figFn = ofigs + figFn + '.png'
  matplotlib.pylab.savefig(figFn,format=format,dpi=dpi)

  with open(latexFn, 'a') as f:
    f.write('\n \\begin{frame} \n')
    f.write('\n \\frametitle{ \n')
    f.write(figCaption)
    f.write('}')

  LatexFig(latexFn=latexFn, figFn=figFn, figLabel=figFn,
           figCaption=figCaption, skipCaption=skipCaption,
           skipLabel=skipLabel)
  with open(latexFn, 'a') as f:
      f.write('\n \\end{frame} \n')

def Excise(filename, start, end):

  with open(filename) as infile, open(filename + ".out666", "w") as outfile:

    for line in infile:
      if line.strip() == start:
        print('zereshk')
        break
      outfile.write(line)

    for line in infile:
      if line.strip() == end:
        print('albaloo!')
        break
    for line in infile:
      outfile.write(line)

  os.remove(filename)
  os.rename(filename + ".out", filename)


def PltCloseStat(stat=True):
  if stat:
    plt.close()

  return None

def SaveFigStat(stat=False, figCaption='test caption'):

  if stat:
    SaveFig_addSlide(
        figCaption=figCaption,
        latexFn=latexInputFn, ofigs=ofigs,
        figFn='', format='png', dpi=200)

  return None

### this function adds a figure with caption and label to a latex file
def DfLatexTable(df):

  table = ''
  n = len(df)
  colNames = df.columns
  top = ' & '.join(colNames)
  table = top + ' \\\\' + '  \n' + '    '
  table = table + '\hline' + '\n'
  table = table + '\hline' + '\n'
  for i in range(n):
    dfRow = df.iloc[i,:]
    tableRow = ' & '.join(dfRow)
    table = table + tableRow + '\\\\' + '\n' + '    '
    table = table + '\hline' + '\n'

  table = table + '\hline' + '\n'

  return table

## creates a latex file which would create a pdf after running
# the pdf has all the photos found in path/figs/
# the latex files will be created in path with the name latexFn
def ConcatPhotos_viaLatex(path, latexFn): 
    figsPath = path + "/figs/"
    files = os.listdir(figsPath)
    
    fn = path + latexFn
    LatexSkin(fn=fn, docType='article', figsPath='{figs/}', latexInputFn='Input.tex')
    latexFn0 = path + "input.tex"
    for file in files:
        LatexFig(
            latexFn=latexFn0,
            figFn=file,
            figLabel=None, figCaption=None, scale=str(0.5))

"""        
path = "/Users/rz13/Dropbox/Reza_Docs/morgage_application2/rent/"
latexFn = "rent_boa_copies.tex"
ConcatPhotos_viaLatex(path=path, latexFn=latexFn)
"""

