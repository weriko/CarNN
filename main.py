
import pygame



import random

#Approach 1: Sensors are just the distance to every wall
#Approach 2: Sensors do measure distance if they can detect wall
#TODO ADD RANDOM CARS THAT DONT WORK WITH AI FOR VARIATION
#TODO JUST GET THE DISTANCE AND ANGLE OF THE WALL AND USE 2 INPUT NEURONS 
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Set up the drawing window
screen_size = [1300,1300]
import math


import numpy as np


#SCORE WEIGHT:
    # CHECKPOINT - 50P (need to substract time to reach soonTM)
    # EVERY FRAME ALIVE - 0.02P
    # 


class Car(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.movex = 0 # move along X
        self.movey = 0 # move along Y
        self.frame = 0 # count frames
        self.speedx = random.randint(1,4)*random.choice([1,-1])
        self.speedy = random.randint(1,4)*random.choice([1,-1])
        self.image = pygame.image.load("carrito.png")
        self.rect = self.image.get_rect()
        self.checkpoint_flags = []
        self.has_NN = False #Checks if nn is already trained
        self.sensor_logs = []
        
        self.score = 0
        
        middlex = int((self.rect.left-self.rect.right)/2)
        middley = int((self.rect.top-self.rect.bottom)/2)
        
        self.sensors = (
        pygame.Rect([-25+middlex, -25+middley,40,40]),
        pygame.Rect([25+middlex, 25+middley,40,40]),
        pygame.Rect([25+middlex,-25+middley,40,40]), 
        pygame.Rect([-25+middlex,25+middley,40,40]),
        
        pygame.Rect([-40+middlex, -0+middley,30,30]),
        pygame.Rect([40+middlex, -0+middley,30,30]),
        pygame.Rect([0+middlex, -40+middley,30,30]),
        pygame.Rect([0+middlex, 40+middley,30,30]),
       
        
        ) #I have to fix this pos zzzz
        self.speed_logs = []
        
        self.sensors_values = [(0,0),
                               (0,0),
                               (0,0),
                               (0,0),
                               (0,0),
                               (0,0),
                               (0,0),
                               (0,0),]
        print(self.rect.bottom)
    def train(self,num=0):
        """
        
        x = np.true_divide(np.array(self.sensor_logs),10)
        y = np.true_divide(np.array(self.speed_logs),10) #Cheap regularization
        """
        x = np.array(self.sensor_logs)
        y = np.array(self.speed_logs)
        self.NN.compile(loss='mean_squared_error', optimizer='adam')
        self.NN.fit(x,y,epochs=5)
        self.NN.save(f'NeuralNetworks/model{str(num)}.h5')
        
 
 
    def accelerate(self):
        to_add=[]
        for i in self.sensors_values:
            to_add.extend(i)
        to_add.extend([self.score])
            
        try:
            if self.has_NN:
                to_add = np.array(to_add)
                to_add = np.expand_dims(to_add,0)
                
                p = self.NN.predict(to_add.reshape(1,17), )
                self.speedx = p[0][0]*random.randint(1,5)
                self.speedy = p[0][1]*random.randint(1,5)
                if abs(p[0][0])<=0.1 and abs(p[0][1])<0.1:
                    self.speedx = random.randint(1,8)*random.choice([1,-1])
                    self.speedy = random.randint(1,8)*random.choice([1,-1])
                else:
                    print(p)
                
            else:
                self.speedx = random.randint(1,8)*random.choice([1,-1])
                self.speedy = random.randint(1,8)*random.choice([1,-1])
                
        except Exception as e:
            print(e)
            
            self.speedx = random.randint(1,8)*random.choice([1,-1])
            self.speedy = random.randint(1,8)*random.choice([1,-1])
        
    def move(self):
        """
        Update sprite position
        """
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        for i in self.sensors:
            i[0]+=self.speedx
            i[1]+=self.speedy
            
            
    def move_ip(self,pos):
        self.rect.x = pos[0]
        self.rect.y=pos[1]
        for i in self.sensors:
            i[0]+=pos[0]
            i[1]+=pos[1]
        
    
    def destroy(self):
        pass
        




class Game:
    def __init__(self):
        self.running = True
        self.cars = []
        self.out_cars = []
        self.screen = pygame.display.set_mode(screen_size)
        self.draw_track()
        self.top_models = []
        self.frame = 1
        for i in range(5):
          
            try:     
                self.top_models.append(tf.keras.models.load_model(f'NeuralNetworks/model{i}.h5'))
             
            except Exception as e:
                print(e)
                
                break
        
        self.model_initialized = self.top_models !=[]
        
        
        
        
        
    def gen_cars(self,x,rand=True):
        for i in range(x):
       
            self.cars.append(Car())
            
      
            self.cars[i].move_ip((random.randint(50,150),   random.randint(50,100  )))
            self.cars[i].checkpoint_flags = [0]*len(self.checkpoints)
            if self.model_initialized:
                self.cars[i].NN = random.choice(self.top_models)
                self.cars[i].has_NN = True
                print("A")
            else:
                self.cars[i].NN = Sequential()
                self.cars[i].NN(tf.keras.Input(shape=(17,)))

                self.cars[i].NN.add(Dense(17, activation="relu"))
                self.cars[i].NN.add(Dense(4, activation="tanh"))
                self.cars[i].NN.add(Dense(2, activation="linear"))
                self.cars[i].NN.build((2,17,))
            
           
        
        
    def move_car(self,carrito):   
        flag = True
        carrito.move()
        try:
            """
            if carrito.rect.left<0 or carrito.rect.right>screen_size[0]:
                self.cars.remove(carrito)
                carrito.destroy()
                
            
            elif carrito.rect.top<0 or carrito.rect.bottom>screen_size[1]:
                self.cars.remove(carrito)
            """
            collide_checkpoint = carrito.rect.collidelist(self.checkpoints) #Calculate checkpoint score
            if collide_checkpoint !=-1:
                if not carrito.checkpoint_flags[collide_checkpoint]:
                    carrito.score+=60
                    carrito.checkpoint_flags[collide_checkpoint] = 1
                
                
            if carrito.rect.collidelist(self.track)!= -1:
                flag = False
                carrito.score -= 50
                self.out_cars.append(carrito)
                self.cars.remove(carrito)
                
                
            any_flag = False
            
            
            for i in range(len(carrito.sensors)): #This checks all collisions on sensors to get distance from the car to the wall
                
                temp =carrito.sensors[i].collidelist(self.track)
                
                if temp!=-1: #IF COLLISION DETECTED, IT CREATES A RECTANGLE THAT OVERLAPS BOTH RECTS, AND SAVES THE x AND y VALUES IN THE SENSOR VALUES
                    any_flag = True
                    temp = carrito.sensors[i].clip(self.track[temp])
                    carrito.sensors_values[i] = (abs(carrito.sensors[i].x-temp.x),
                                                 abs(carrito.sensors[i].y-  temp.y))
            if any_flag:
                to_add=[]
                for i in carrito.sensors_values:
                    to_add.extend(i)
                
                
                carrito.sensor_logs.append( to_add+[carrito.score])
                
                carrito.speed_logs.append((carrito.speedx,carrito.speedy))
                
                    #THIS LOGS THE SENSOR VALUES, PLUS 
                    
                    #print(carrito.sensors_values[i],i)
                #pygame.draw.rect(self.screen, [0,255,0],carrito.sensors[i],0)
            if flag:
                carrito.accelerate()
                carrito.score+=0.001
                
            
           
        except Exception as e:
            print(e)
            print(carrito)
            print(self.cars)
            pass
        
    def draw_track(self):
        
        self.checkpoints = []
        self.screen.fill((255, 255, 255))
        self.track = [
        [0,0,1300,15] ,   
            
        [0,0,15,1300],
        [200,0,15,1100],
        [200,1100,600,15],
        [800,200,15,1000],
        [500,200,400,15],
        [1000,0,15,1300],
        [300,0,15,1000], #Vertical line
        [300,1000,400,15], #Horizontal line
        [500,200,15,600]]
        
        self.checkpoints.append([15,500,185,15])
        self.checkpoints.append([15,150,185,15])
        self.checkpoints.append([15,350,185,15])
        self.checkpoints.append([15,125,185,15])
        self.checkpoints.append([15,200,185,15])
        self.checkpoints.append([15,250,185,15])
        self.checkpoints.append([15,300,185,15])
        
        self.checkpoints.append([15,1000,185,15])
        self.checkpoints.append([200,1115,15,185])
        self.checkpoints.append([800,1200,15,185])
        self.checkpoints.append([815,500,185,15])
        self.checkpoints.append([900,200,100,15])
        
        
        for i in self.track:
                pygame.draw.rect(self.screen, [255,0,0],i,0)
        for i in self.checkpoints:
                pygame.draw.rect(self.screen, [0,255,0],i,0)
        pygame.display.flip()
            
            
    
    def calculate_scores(self,n_scores):
        self.out_cars.sort(key=lambda x: x.score, reverse=True)
        for i in self.out_cars[:n_scores]:
            print(i.score)
        for i in range(len(self.out_cars[:n_scores])):
            self.out_cars[i].train(num=i)
        f = open("LOGS","w")
        logs = ""
        for i in self.out_cars[1].sensor_logs:
            for p in i:
                logs+= f"{str(p)} "
            logs+="\n"
        f.write(logs)
        f.close()
        self.out_cars[1].NN.summary()
        
        
        
        
    def run(self):
        flag = True
        
        
        pygame.init()
        pygame.time
        while self.running:
            self.frame+=1
            pygame.time.delay(10)
        
            # Did the user click the window close button?
            
            for event in pygame.event.get():
        
                if event.type == pygame.QUIT:
        
                    self.running = False
        
            
           
        
            self.screen.fill((255, 255, 255))
            for i in self.cars:
                self.move_car(i)
                self.screen.blit(i.image,i.rect)
            for i in self.track:
                pygame.draw.rect(self.screen, [255,0,0],i,0)
            for i in self.checkpoints:
                
                pygame.draw.rect(self.screen, [0,255,0],i,0)
            
            pygame.display.update()
            
            if self.cars ==[] and flag:
                flag = False
                self.calculate_scores(5)
        
        
            
        
        
        # Done! Time to quit.
        
        pygame.quit()
game = Game()

game.gen_cars(10)
game.run()