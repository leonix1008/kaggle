import openpyxl

wb = openpyxl.load_workbook('train.xlsx', read_only = True)
ws = wb['train']

def getrawlist(searchstring):
    rawlist = []
    for column in ws[searchstring]:
        for cell in column:
            rawlist.append(cell.value)
    return rawlist

def returncleanlist(rawlist):
    count = 0
    for val in rawlist:
        if val == None:
            count += 1
    percentage = (100 / len(rawlist)) * count
    if percentage > 20:
        rawlist = []
    else:
        pass
    return rawlist

def deleteemptylist(rawlist):
    if bool(rawlist) == False:
        del rawlist

def returnliststoworkwith():
    rawpassengeridlist = getrawlist('A2:A892')
    rawsurvivedlist = getrawlist('B2:B892')
    rawpclasslist = getrawlist('C2:C892')
    rawnamelist = getrawlist('D2:D892')
    rawsexlist = getrawlist('E2:E892')
    rawagelist = getrawlist('F2:F892')
    rawsibsplist = getrawlist('G2:G892')
    rawparchlist = getrawlist('H2:H892')
    rawticketlist = getrawlist('I2:I892')
    rawfarelist = getrawlist('J2:J892')
    rawcabinlist = getrawlist('K2:K892')
    rawembarkedlist = getrawlist('L2:L892')
    passengeridlist = returncleanlist(rawpassengeridlist)
    survivedlist = returncleanlist(rawsurvivedlist)
    pclasslist = returncleanlist(rawpclasslist)
    namelist = returncleanlist(rawnamelist)
    sexlist = returncleanlist(rawsexlist)
    agelist = returncleanlist(rawagelist)
    sibsplist = returncleanlist(rawsibsplist)
    parchlist = returncleanlist(rawparchlist)
    ticketlist = returncleanlist(rawticketlist)
    farelist = returncleanlist(rawfarelist)
    cabinlist = returncleanlist(rawcabinlist)
    embarkedlist = returncleanlist(rawembarkedlist)
    
    return passengeridlist, survivedlist, pclasslist, namelist, sexlist, agelist, sibsplist, parchlist, ticketlist, farelist, cabinlist, embarkedlist

passengeridlist, survivedlist, pclasslist, namelist, sexlist, agelist, sibsplist, parchlist, ticketlist, farelist, cabinlist, embarkedlist = returnliststoworkwith()