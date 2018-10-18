import numpy as np
import cv2
import math

def HomographyMatrix(src_img_pts,world_plane_pts):
    
    tb=[]
    ta=[]
    
    for i in range(0,len(src_img_pts)):
        
       tmp=[world_plane_pts[i][0],world_plane_pts[i][1],1,0,0,0,-src_img_pts[i][0]*world_plane_pts[i][0],-src_img_pts[i][0]*world_plane_pts[i][1]]            
       ta.append(tmp)
       tmp=[0,0,0,world_plane_pts[i][0],world_plane_pts[i][1],1,-src_img_pts[i][1]*world_plane_pts[i][0],-src_img_pts[i][1]*world_plane_pts[i][1]]
       ta.append(tmp)
       A=np.asarray(ta)
       
       tmp=[src_img_pts[i][0],src_img_pts[i][1]]
       tb.append(tmp)
       tmp=np.asarray(tb)
       b=tmp.reshape(-1)
       
       
    tmp=np.dot(np.linalg.pinv(A),b.transpose())
    H=np.zeros((3,3)) 
    H[0]=tmp[0:3]
    H[1]=tmp[3:6]
    H[2][0:2]=tmp[6:8]
    H[2][2]=1 
    
    return H

def WeightedAverageRGBPixelValue(pt, img):
    
    x1=int(math.floor(pt[0]))
    x2=int(math.ceil(pt[0]))
    y1=int(math.floor(pt[1]))
    y2=int(math.ceil(pt[1]))
        
    Wp=1/np.linalg.norm(np.array([pt[0]-x1,pt[1]-y1]))
    Wq=1/np.linalg.norm(np.array([pt[0]-x1,pt[1]-y2]))
    Wr=1/np.linalg.norm(np.array([pt[0]-x2,pt[1]-y1]))
    Ws=1/np.linalg.norm(np.array([pt[0]-x2,pt[1]-y2]))
    
    pixel_value = (Wp*img[y1][x1] + Wq*img[y2][x1] + Wr*img[y1][x2] + Ws*img[y2][x2])/(Wp+Wq+Wr+Ws)
    
    pixel_value
    return pixel_value

def ProjectionImage(H,src_img,world_plane_img):
    
    for i in range(0,np.shape(world_plane_img)[0]-1):
        for j in range(0,np.shape(world_plane_img)[1]-1):
            tmp=np.array([j,i,1])
            xp=np.array(np.dot(H,tmp))
            xp=xp/xp[2]
            if((xp[0]>0)and(xp[0]<src_img.shape[1]-1)and(xp[1]>0)and(xp[1]<src_img.shape[0]-1)):
                world_plane_img[i][j]=WeightedAverageRGBPixelValue(xp,src_img)
            
        
         
    output_img = world_plane_img
    return output_img    

# Main Function Begins 

image1=cv2.imread("1.jpg")
image2=cv2.imread("2.jpg")
image3=cv2.imread("3.jpg")
imageJ=cv2.imread("Jackie.jpg")

t1=[]
t2=[]
t3=[]
tJ=[]

tJ.append([0.0,0.0,1.0])
tJ.append([1279.0,0.0,1.0])
tJ.append([0.0,719.0,1.0])
tJ.append([1279.0,719.0,1.0])

pts_imageJ=np.asarray(tJ)

t1.append([1509.0,157.0,1.0])
t1.append([2971.0,727.0,1.0])
t1.append([1481.0,2257.0,1.0])
t1.append([3017.0,2061.0,1.0])

pts_image1=np.asarray(t1)

t2.append([1309.0,317.0,1.0])
t2.append([3021.0,593.0,1.0])
t2.append([1295.0,2027.0,1.0])
t2.append([3033.0,1899.0,1.0])

pts_image2=np.asarray(t2)

t3.append([913.0,728.0,1.0])
t3.append([2802.0,377.0,1.0])
t3.append([889.0,2093.0,1.0])
t3.append([2854.0,2238.0,1.0])

pts_image3=np.asarray(t3)


# if size does not match exit(1) condition to be included for src and world pts sizes ?

H=HomographyMatrix(pts_imageJ,pts_image1)
output=ProjectionImage(H,imageJ,image1)
cv2.imwrite('JackieOutput_image1.jpg',output)

H=HomographyMatrix(pts_imageJ,pts_image2)
output=ProjectionImage(H,imageJ,image2)
cv2.imwrite('JackieOutput_image2.jpg',output)

H=HomographyMatrix(pts_imageJ,pts_image3)
output=ProjectionImage(H,imageJ,image3)
cv2.imwrite('JackieOutput_image3.jpg',output)



















