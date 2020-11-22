
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
    # TOTAL DISTANCE TRAVELED 0.002 per pixel


class Car(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.total_movex = []
        self.total_movey = [] 
        self.frame = 0 # count frames
        self.speedx = random.randint(1,4)*random.choice([1,-1])
        self.speedy = random.randint(1,4)*random.choice([1,-1])
        self.image = pygame.image.load("carrito.png")
        self.rect = self.image.get_rect()
        self.checkpoint_flags = []
        self.has_NN = False #Checks if nn is already trained
        self.sensor_logs = []
        self.scores = []
        
        self.score = 0
        self.sensor_values = []
        
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
        
        self.sensors_values = [0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,]
        #print(self.rect.bottom)
    def train(self,num=0):
        """
        
        x = np.true_divide(np.array(self.sensor_logs),10)
        y = np.true_divide(np.array(self.speed_logs),10) #Cheap regularization
        """
      
        
        to_add = self.sensor_logs
       
        to_add = np.c_[to_add,np.array(self.scores)]
        to_add =np.c_[to_add,np.array(self.total_movex)]
        to_add =np.c_[to_add,np.array(self.total_movey)]
       
        x = np.array(to_add)
        y = np.array(self.speed_logs)
        self.NN.compile(loss='mean_squared_error', optimizer='adam')
        self.NN.fit(x,y,epochs=5)
        self.NN.save(f'NeuralNetworks/model{str(num)}.h5')
        
 
 
    def accelerate(self):
        to_add = np.array(self.sensors_values)
        
        
        to_add = np.r_[to_add,np.array(self.score)]
        to_add = np.r_[to_add,np.array(self.total_movex[-1])]
        to_add = np.r_[to_add,np.array(self.total_movey[-1])]
        
     
        
   
        try:
            if self.has_NN:
          
                to_add = np.expand_dims(to_add,0)
                
                p = self.NN.predict(to_add.reshape(1,19), )
                self.speedx = p[0][0]*random.randint(1,5)
                self.speedy = p[0][1]*random.randint(1,5)
                if abs(p[0][0])<=0.01 and abs(p[0][1])<0.01:
                    self.speedx = random.randint(1,8)*random.choice([1,-1])
                    self.speedy = random.randint(1,8)*random.choice([1,-1])
                else:
                    pass
                
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
        self.total_movex.append(abs(self.speedx))
        self.total_movey.append((self.speedy))
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
        self.end_sess = False
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
                try:
                    f = open("SCORES","r")
                    total = [float(x) for x in f.read().split("\n") if x!=""]
                    f.close()
                    
                    choice = np.random.choice(range(0,7), p=[x/sum(total) for x in total])
                    self.cars[i].NN = self.top_models[choice]
                    self.cars[i].has_NN = True
                except Exception as e:
                    print(e)
                    
                    self.cars[i].NN = random.choice(self.top_models)
                    self.cars[i].has_NN = True
                    
             
            else:
                self.cars[i].NN = Sequential()
                self.cars[i].NN(tf.keras.Input(shape=(19,)))

                self.cars[i].NN.add(Dense(19, activation="relu"))
                self.cars[i].NN.add(Dense(4, activation="tanh"))
                self.cars[i].NN.add(Dense(2, activation="linear"))
                self.cars[i].NN.build((2,19,))
        if rand:
            for i in range(int(x/2)):
                
               
                
               car = Car()
      
               car.move_ip((random.randint(50,150),   random.randint(50,100  )))
               car.checkpoint_flags= [0]*len(self.checkpoints)
               car.NN = Sequential()
               car.NN(tf.keras.Input(shape=(19,)))

               car.NN.add(Dense(19, activation="relu"))
               car.NN.add(Dense(4, activation="tanh"))
               car.NN.add(Dense(2, activation="linear"))
               car.NN.build((2,19,))
               self.cars.append(car)
        
                
            
           
        
        
    def move_car(self,carrito):   
        flag = True #THIS FLAG CHECKS IF THE CAR IS NOT DESTROYED
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
                    carrito.score+=100*collide_checkpoint
                    carrito.checkpoint_flags[collide_checkpoint] = 1
                
                
            if carrito.rect.collidelist(self.track)!= -1:
                flag = False
                carrito.score -= 100
                self.out_cars.append(carrito)
                self.cars.remove(carrito)
                
                
            any_flag = False
            
            
            for i in range(len(carrito.sensors)): #This checks all collisions on sensors to get distance from the car to the wall
                
                temp =carrito.sensors[i].collidelist(self.track)
                
                if temp!=-1: #IF COLLISION DETECTED, IT CREATES A RECTANGLE THAT OVERLAPS BOTH RECTS, AND SAVES THE x AND y VALUES IN THE SENSOR VALUES
                    any_flag = True
                    
                    temp = carrito.sensors[i].clip(self.track[temp])
                    carrito.sensors_values[i] = abs(carrito.sensors[i].x-temp.x)
                    carrito.sensors_values[i+1]=    abs(carrito.sensors[i].y-  temp.y)
                    
           
           
            
            
            carrito.sensor_logs.append( carrito.sensors_values)
            
            carrito.speed_logs.append((carrito.speedx,carrito.speedy))
            
                    #THIS LOGS THE SENSOR VALUES, PLUS 
                    
                    #print(carrito.sensors_values[i],i)
                #pygame.draw.rect(self.screen, [0,255,0],carrito.sensors[i],0)
            if flag:
                carrito.accelerate()
                carrito.score+=0.0001
                carrito.score +=0.0005*carrito.speedx  #ADDS SCORE FOR BEING FAST
                carrito.score +=0.0005*carrito.speedy
                carrito.scores.append(carrito.score)
            else:
                carrito.scores.append(carrito.score)
                
            
           
        except Exception as e:
            print(e)
            #print(carrito)
            #print(self.cars)
            pass
        
    def draw_track(self):
        
        
        self.screen.fill((255, 255, 255))
        self.track = [
        [0,0,1300,15] ,   
            
        [0,0,15,1300],
        
        
        [0,1285,1300,15],
        
        [200,0,15,1100],
        [200,1100,600,15],
        [800,200,15,1000],
        [500,200,400,15],
        [1000,0,15,1300],
        [300,0,15,1000], #Vertical line
        [300,1000,400,15], #Horizontal line
        [500,200,15,600]]
        
        self.checkpoints = [
        
        [15,125,185,15],
        [15,150,185,15],
        [15,200,185,15],
        [15,250,185,15],
        [15,300,185,15],
        [15,350,185,15],
        [15,400,185,15],
        [15,500,185,15],
        [15,600,185,15],
        [15,700,185,15],
        
        
   
        [15,1000,185,15],
        [200,1115,15,185],
        [800,1200,15,185],
        [815,500,185,15],
        [900,200,100,15],
        ]
        
        
        for i in self.track:
                pygame.draw.rect(self.screen, [255,0,0],i,0)
        for i in self.checkpoints:
                pygame.draw.rect(self.screen, [0,255,0],i,0)
        pygame.display.flip()
            
            
    
    def calculate_scores(self,n_scores):
        self.out_cars.sort(key=lambda x: x.score, reverse=True)
        scores_log = ""
        for i in self.out_cars[:n_scores]:
            print(i.score)
            scores_log+=f"{i.score}\n"
        
            
            
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
        f = open("SCORES","w")
        f.write(scores_log)
        f.close()
        
        
        self.out_cars[1].NN.summary()
        
        
        
        
    def run(self):
        flag = True
        
        
        pygame.init()
        pygame.time
        while self.running:
            self.frame+=1
            pygame.time.delay(2)
        
            # Did the user click the window close button?
            
            for event in pygame.event.get():
        
                if event.type == pygame.QUIT:
        
                    self.running = False
                    self.end_sess=True
        
            
           
        
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
                self.calculate_scores(7)
                self.running = False
        
        
            
        
        
        # Done! Time to quit.
        
        pygame.quit()
while True:
    game = Game()
    
    game.gen_cars(60, rand=False)
    game.run()
    if game.end_sess:
        break