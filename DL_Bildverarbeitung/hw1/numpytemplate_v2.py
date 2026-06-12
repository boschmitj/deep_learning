import numpy as np

#define a random generator with a seed here using numpy, using the newer convention for numpy https://numpy.org/doc/stable/reference/random/generator.html
rng = np.random.default_rng(seed=42)


##################################################
# part 1 the easy stuff, indexing, searching
##################################################

a = np.arange(36).reshape((6, 6))
print(a)
def index1(mat):
    #Return row indexed with 5 (and columns 0,1)
    return mat[5, :2]

res = index1(a)
print(res)

def index2(mat):
    #Return column indexed with 2 (and rows 0,1)
    return mat[:2, 5]   

res = index2(a)
print(res)

def index3(mat):
    #Return the columns indexed with 1 and with 3 (and rows 0,1)
    return mat[:2, [1, 3]] # oder expliziter mit np.array([1, 3])

res = index3(a)
print(res)
print(a)
def index4(mat):
    #Return the rows indexed with 0 to 2 (and columns 0,1). It should be 3 rows here
    return mat[0:3, :2]

res = index4(a)
print(res)

def index5(mat):
    # return the last row using a negative index (and columns 0,1)
    return mat[-1, :2]

res = index5(a)
print(res)

def index6(mat): 
    # return the third last row using a negative index (and columns 0,1)
    return mat[-3, :2]

res = index6(a)
print(res)

def index7(mat):
    # return the the last two rows using a negative start index (and columns 0,1)
    return mat[-2:, :2]

res = index7(a)
print(res)

def index8(mat):
    # return the the last three columns using a negative start index (and rows 0,1)
    return mat[:2, -3:]

res = index8(a)
print(res)

def index9(mat):
    # return the columns indexed with 3,4 using a negative start index and a negative stop index (and rows 0,1)
    return mat[-3:-1, :2]

res = index9(a)
print(res)

def index10(mat):
    # return the columns indexed with 0,1,2 using a negative start index and a negative stop index (and rows 0,1)
    return mat[-6:-3, :2]

res = index10(a)
print(res)

print(a)
def index11(mat):
    # return every 2nd column, starting at index 0  (and rows 0,1)
    return mat[::2, :2]   

res = index11(a)
print(res)

def index12(mat):
    # return every 3rd column, starting at index 1 (and rows 0,1)
    return mat[1::3, :2]

res = index12(a)
print(res)

def index13(mat):
    # return every 2nd column, starting at the last index, in reversed order  (and rows 0,1)
    return mat[-1::-2, :2]

res = index13(a)
print(res)

def index14(mat):
    # return every 2nd column, starting at the second last index, in reversed order (and rows 0,1) 
    return mat[-2::-2, :2] 
 
res = index14(a)
print(res) 

print(a)
b= np.arange(0,22)
def index15(mat):
    # return every 3rd element, between index 3 and 12 (12 not included)
    return b[3:12:3]

res = index15(a)
print(res)
      
a = rng.integers(low=2,high=15,size=(5,5))
print(a)
def indexr1(mat):
    #Return a matrix which is true where mat-values are higher than 5  (and false otherwise) 
    return mat > 5

res = indexr1(a)
print(res)

def indexr2(mat):
    #Return the indices of the matrix where mat-values are higher than 5   
    # return np.where(mat > 5)
    # bessere lesbarkeit
    # return list(zip(*np.where(mat > 5))) 
    # oder
    return np.argwhere(mat > 5) # für direkte ausgabe als narrays

res = indexr2(a)
print(res)

print(a)
def indexr3(mat):
    #Return the values where mat-values are higher than 5 
    #will the result be a matrix?  
    return mat[mat > 5]
    # No, it won't be, instead its shape will be (19,1) as its flattened
    

res = indexr3(a)
print(res)
print(res.shape)
# Für Ausgabe als Spaltenvektor
print(res.reshape(-1, 1))

def indexr4(mat):
    #Return the values where mat-values are higher than 5 and lower than 10
    #will the result be a matrix?  
    return mat[(mat > 5) & (mat < 10)]
    # No again, same reasoning

res = indexr4(a)
print(res)

##################################################
# part 2 the easy stuff,  element-wise operations
##################################################

a = np.arange(9).reshape((3, 3))+1
print(a)
def op1(mat):
    # return the sum over all elements    
    return np.sum(mat)

res = op1(a)
print(res)

a = np.arange(36).reshape((3, 3, 4))    
print(a)  
def op2(mat):
    # return the sum over the axis #1 , indexing starts at 0
    return np.sum(mat[0:], axis=1) 
    # Addiere jeweils die Zeilen eines Blocks (vertikal), 
    # axis = 2 wäre die Spalten (horizontal) und axios = 0 wäre die Blöcke an sich (übereinander legen)


res = op2(a)
print(res)
     
a = np.arange(9).reshape((3, 3))
print(a)
def op3(mat):
    # return a scaled version so that it sums up to 1  
    total = np.sum(mat)
    return mat / total
  

res = op3(a)
print(res)
print(np.sum(res))

def op4(mat):
    # square each entry element wise  
    return mat ** 2

res = op4(a)
print(res)    

a = np.arange(6).reshape((3, 2))
b = np.arange(6).reshape((3, 2)) -5  
def op5(mat1,mat2):
    # multiply both element-wise
    return a * b

print(a)
print(b)
res = op5(a,b)
print(res)

##################################################
# part 5 the intermediate stuff, inner products and the like
##################################################    

a = np.arange(6).reshape((3, 2))
b = np.arange(3) -5 

def op6(mat1,v):
    # compute the inner product between v and each vector in mat1 which is defined by fixing index in axis #1 and cycling through elements in axis #0
    # vertikal
    # return list(np.inner(v, mat1[:, i]) for i in range(mat1.shape[1]))
    # oder 
    return mat1.T @ v


res = op6(a,b)
print(res)

a = np.arange(12).reshape((4, 3))
b = np.arange(3) -5 

def op7(mat1,v):
    # compute the inner product between v and each vector in mat1 which is defined by fixing index in axis #0 and cycling through elements in axis #1
    # horizontal
    return list(np.inner(v, mat1[i, :]) for i in range(mat1.shape[0]))

print("op7")
print(a)
print(b)
res = op7(a,b)
print(res)        
  
a = np.arange(6).reshape((3, 2))
b = np.arange(3) -5
def op8(mat1,v):
    # matrix multiply v from the left to mat1
    # answer: is this equal to op6 or op7 ?
    return v @ mat1
    # Yes, does the same

res = op8(a,b)
print(res)

a = np.arange(72).reshape((4, 3,3,2))
b = np.arange(3) -5 

def op9(mat1,v):
    # compute the inner product between v and each vector in mat1 which is defined by fixing index in axis #0 and #1 and #3 and cycling through elements in axis #2
    result = []
    a = mat1.shape[0]
    b = mat1.shape[1]
    d = mat1.shape[3]
    for i in range(a):
        for j in range(b):
            for l in range(d):
                # hier inner product über k --> axis = 2
                result.append(np.inner(v, mat1[i, j, :, l]))
    return result
        
print(a)
res = op9(a,b)
print(res)  
print(len(res)) # richtigerweise 24 ergebnisse
     
