import numpy as np
import utils

class i2b2Line:
    def __init__(self, x, s, start, end, file_id, line_id):
        self.X=x        
        self.S=s
        self.Start=start
        self.End=end
        self.File_ID=file_id
        self.Line_ID=line_id
        self.Length=len(self.X)

class i2b2Corpus:
    def __init__(self, file_name):
        self.Lines=[]
        for line in open(file_name, 'r'):
            if line[0]=='#': continue
            segs=line.split(' ')
            file_id=segs[0]
            line_id=segs[1]
            line_data=[seg.split(':') for seg in segs[2:]]
            X=[int(v[0]) for v in line_data]
            X.insert(0,-1) # insert a placeholder for line_start
            S=[int(v[1]) for v in line_data]
            S.insert(0,0) # insert line_start token
            Start=[int(v[3]) for v in line_data]
            Start.insert(0,0)
            End=[int(v[4]) for v in line_data]
            End.insert(0,0)
            l=i2b2Line(X, S, Start, End, file_id, line_id)
            self.Lines.append(l)