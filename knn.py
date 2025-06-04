import heapq

class Point:
    def __init__(self,distance,label):
        self.distance=distance
        self.label=label

    def __lt__(self, other):
        return self.distance < other.distance 
    
class knn:
    def __init__(self,x,y,k):
        self.X_train=x
        self.Y_train=y
        self.k=k
    
    def calculate_distance(self, x1, x2):
        return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5

    def predict(self,x):
        distances=[] #this is a minheap to store the nearest distances first so that we can pop in O{1} time

        for i in range(len(self.X_train)):
            dist=self.calculate_distance(self.X_train[i],x)
            heapq.heappush(distances, Point(dist,self.Y_train[i]))

        reqd=self.k

        map_of_points={}
        max_recurring=0
        max_label=None
        
        for i in range(reqd):
            point=heapq.heappop(distances)
            if point.label in map_of_points:
                map_of_points[point.label]+=1
                if map_of_points[point.label]>max_recurring:
                    max_recurring=map_of_points[point.label]
                    max_label=point.label
            else:
                map_of_points[point.label]=1
                if map_of_points[point.label]>max_recurring:
                    max_recurring=map_of_points[point.label]
                    max_label=point.label
            
        return max_label
    


        

         


    