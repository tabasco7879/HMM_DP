tag_dict={        
    'LINE_START':(0,0),
    'TEXT':(1,1),
    'DATE-B':(2,2),
    'DATE-I':(3,3),
    'PATIENT-B':(4,4),
    'PATIENT-I':(5,5),
    'DOCTOR-B':(6,4),
    'DOCTOR-I':(7,5),
    'USERNAME-B':(8,4),
    'USERNAME-I':(9,5),
    'AGE-B':(10,6),
    'AGE-I':(11,7),
    'COUNTRY-B':(12,8),
    'COUNTRY-I':(13,9),
    'ORGANIZATION-B':(14,8),
    'ORGANIZATION-I':(15,9),
    'STREET-B':(16,8),
    'STREET-I':(17,9),        
    'CITY-B':(18,8),
    'CITY-I':(19,9),
    'STATE-B':(20,8),
    'STATE-I':(21,9),
    'ZIP-B':(22,8),
    'HOSPITAL-B':(23,8),
    'HOSPITAL-I':(24,9),
    'LOCATION-OTHER-B':(25,8),
    'LOCATION-OTHER-I':(26,9),
    'PROFESSION-B':(27,10),
    'PROFESSION-I':(28,11),
    'PHONE-B':(29,12),
    'PHONE-I':(30,13),
    'FAX-B':(31,12),
    'FAX-I':(32,13),
    'EMAIL-B':(33,12),
    'EMAIL-I':(34,13),        
    'URL-B':(35,12),
    'URL-I':(36,13),
    'IPADDRESS-B':(37,12),
    'IPADDRESS-I':(38,13),
    'MEDICALRECORD-B':(39,14),
    'MEDICALRECORD-I':(40,15),
    'IDNUM-B':(41,14),
    'IDNUM-I':(42,15),
    'DEVICE-B':(43,14),
    'DEVICE-I':(44,15),
    'BIOID-B':(45,14),
    'BIOID-I':(46,15),
    'HEALTHPLAN-B':(47,14),
    'HEALTHPLAN-I':(48,15)
    }
        
def get_tags(token, start, end, tags):   
    touched_tags=[]
    for tag in tags:
        if tag[1]<=start and end<=tag[2]:
            touched_tags.append(tag)
        elif tag[1]<end and end<tag[2]:
            touched_tags.append(tag)
        elif tag[1]<start and start<tag[2]:
            touched_tags.append(tag)
        elif start<tag[1] and tag[2]<end:
            touched_tags.append(tag)
    #if touched_tags: print touched_tags        
    tokens=[]
    pos=start
    for tag in touched_tags:
        if pos<tag[1]:
            tokens.append((token[pos-start:tag[1]-start], pos, tag[1], tag_dict['TEXT']))
            pos=tag[1]
        end0=min(end, tag[2])            
        if pos==tag[1]:            
            tokens.append((token[pos-start:end0-start], pos, end0, tag_dict[tag[0]+'-B']))            
        else:
            tokens.append((token[pos-start:end0-start], pos, end0, tag_dict[tag[0]+'-I']))            
        pos=end0
        
    if pos<end:
        tokens.append((token[pos-start:], pos, end, tag_dict['TEXT']))
    return tokens