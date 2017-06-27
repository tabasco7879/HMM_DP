from lxml import etree
import os
from os import listdir
from nltk.tokenize.stanford import StanfordTokenizer
import parser_util

os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0/bin/java.exe'
        
alter_dict={
    '``':'"',    
    "''":'"'    
    }

rep_dict={
    '>':'&gt;',
    '<':'&lt;',
    '&':'&amp;',
    '"':'&quot;',
    '`':"'" 
}

word_dict={}

training_folder='..\\..\\training-PHI-Gold-Set1'
for f in listdir(training_folder):
    if f.endswith('.xml'): #and f=='220-01.xml':
        print os.path.join(training_folder, f)
        root=etree.parse(os.path.join(training_folder, f))
        s=None
        for text_node in root.xpath('//TEXT'):
            s=text_node.text
            break
            
        if s:
            tags=[]
            for tags_node in root.xpath('//TAGS'):
                for tag_node in tags_node:
                    tags.append((tag_node.attrib['TYPE'], int(tag_node.attrib['start']), int(tag_node.attrib['end'])))
                    #print tag_node.tag, tag_node.attrib['start'], tag_node.attrib['end'], tag_node.attrib['text']
                break
                        
            tokens=StanfordTokenizer('C:/Software/stanford-parser-full-2014-10-26/stanford-parser.jar').tokenize(s) 
            start_pos=0
            for token in tokens:
                token=token.encode('utf-8').replace("\xc2\xa0", " ") \
                    .replace("-LRB-", "(").replace("-RRB-", ")") \
                    .replace("-LSB-", "[").replace("-RSB-", "]") \
                    .replace("-LCB-", "{").replace("-RCB-", "}")                    
                find_token=token
                if find_token in alter_dict:                    
                    find_token=alter_dict[find_token]
                    
                find_token=find_token
                cc=[]
                for c in find_token:                    
                    if c in rep_dict \
                        and s.find(rep_dict[c], start_pos)>=0 \
                        and len(s[start_pos:s.find(rep_dict[c], start_pos)].strip())<10:                        
                        cc.append(rep_dict[c])
                    else:
                        cc.append(c)                
                find_token=''.join(cc)
                               
                if not(s.find(find_token, start_pos)>=0 \
                   and len(s[start_pos:s.find(find_token, start_pos)].strip())<2):                      
                    find_token=token
                
                token_pos=s.find(find_token, start_pos)                
                #print token,token_pos
                assert token_pos>=0, token
                start_pos=token_pos+len(find_token)
                tagged_tokens=parser_util.get_tags(find_token, token_pos, start_pos, tags)                
                if len(tagged_tokens)>1:
                    print find_token, token_pos, start_pos, tagged_tokens                    
                for tagged_token in tagged_tokens:
                    if tagged_token[0] in word_dict:
                        word_dict[tagged_token[0]]+=1
                    else:
                        word_dict[tagged_token[0]]=1
                if find_token.endswith('.'): start_pos-=1
            
            