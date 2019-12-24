import glutils
import sys, random, math
import OpenGL
import OpenGL.GL as gl 
from OpenGL.GL import *
from OpenGL.GL.shaders import * 
import numpy as np
import glfw
import pyautogui
strVS = """
layout(location = 0) in vec3 position;
layout (location = 1) in vec2 inTexcoord;
out vec2 outTexcoord;
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform float a;
uniform float b;
uniform float c;
uniform float scale;
uniform float theta;
void main(){
		mat4 rot1=mat4(vec4(1.0, 0.0,0.0,0),
								vec4(0.0, 1.0,0.0,0),
								vec4(0.0,0.0,1.0,0.0),
								vec4(a,b,c,1.0));
		mat4 rot2=mat4(vec4(scale, 0.0,0.0,0.0),
										vec4(0.0, scale,0.0,0.0),
										vec4(0.0,0.0,scale,0.0),
										vec4(0.0,0.0,0.0,1.0));
		mat4 rot3=mat4( vec4(0.5+0.5*cos(theta),  0.5-0.5*cos(theta), -0.707106781*sin(theta), 0),
								vec4(0.5-0.5*cos(theta),0.5+0.5*cos(theta), 0.707106781*sin(theta),0),
								vec4(0.707106781*sin(theta), -0.707106781*sin(theta),cos(theta), 0.0),
								vec4(0.0,				 0.0,0.0, 1.0));
		gl_Position=uPMatrix * uMVMatrix  *rot1*rot2 *rot3 * vec4(position.x, position.y, position.z, 1.0);
		outTexcoord = inTexcoord;
		}
"""
strFS = """
out vec4 FragColor;
in vec2 outTexcoord;
uniform sampler2D GL_TEXTURE0;
void main(){
		FragColor = texture(GL_TEXTURE0, outTexcoord);
		}
"""
screen_size=(1920,1080)
PI = 3.14159265358979323846264
pMatrix=[]
mvMatrix=[]
surface_vertices=[-0.5,-0.5,0, 0.5,-0.5,0, 0.5,0.5,0, 0.5,0.5,0, -0.5,0.5,0, -0.5,-0.5,0]
surface_tex_ma=[0,0, 1,0, 1,1, 1,1, 0,1, 0,0, 0,0, 1,0, 1,1, 1,1, 0,1, 0,0]
surface_vertices2=[-21, 0, -3,21, 0,-3,21, 0, 6,21, 0, 6,-21, 0, 6,-21, 0,-3]
surface_vertices3=[0, 21, 6,0, 21, -3,0, -21, -3,0, -21, -3,0, -21, 6,0, 21, 6]	
cube_vertices=[-0.5,-0.5,-0.5, 0.5,-0.5,-0.5, 0.5,0.5,-0.5, 0.5,0.5,-0.5, -0.5,0.5,-0.5, -0.5,-0.5,-0.5, -0.5,-0.5,0.5,0.5,-0.5,0.5,0.5, 0.5, 0.5,0.5, 0.5, 0.5,-0.5, 0.5, 0.5,-0.5, -0.5, 0.5,-0.5, 0.5, 0.5, -0.5, 0.5, -0.5,-0.5, -0.5, -0.5,-0.5, -0.5, -0.5,-0.5, -0.5, 0.5,-0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5,0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5,0.5, -0.5, 0.5,-0.5, -0.5, 0.5,-0.5, -0.5, -0.5,-0.5, 0.5, -0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5,-0.5, 0.5,-0.5]
cube_tex_ma=[0,0, 1,0, 1,1, 1,1, 0,1, 0,0,0,0, 1,0, 1,1, 1,1, 0,1, 0,0, 1,0, 1,1, 0,1, 0,1, 0,0, 1,0, 1,0, 1,1, 0,1, 0,1, 0,0, 1,0,  0,1, 1,1, 1,0, 1,0, 0,0, 0,1, 0,1, 1,1, 1,0, 1,0, 0,0, 0,1]
things=[]
activate_things=[]
height_dic={}
class thing():
	def __init__(self,vertices,tex_ma,tex_id,a,b,c,scale,r):
		# load shaders
		self.program = glutils.loadShaders(strVS, strFS)
		glUseProgram(self.program)
		# attributes
		self.vertIndex=glGetAttribLocation(self.program, b"position")
		self.texIndex=glGetAttribLocation(self.program, b"inTexcoord")
		self.vertices=vertices
		self.tex_ma=tex_ma
		self.tex_id=tex_id
		self.a=a
		self.b=b
		self.c=c
		self.scale=scale
		self.r=r
		# set up vertex array object (VAO)
		self.vao = glGenVertexArrays(1)
		glBindVertexArray(self.vao)
		# set up VBOs
		vertexData = np.array(self.vertices, np.float32)
		self.vertexBuffer = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
		glBufferData(GL_ARRAY_BUFFER, 4*len(vertexData), vertexData, GL_STATIC_DRAW)
		tcData = np.array(self.tex_ma, np.float32)
		self.tcBuffer = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.tcBuffer)
		glBufferData(GL_ARRAY_BUFFER, 4*len(tcData), tcData,GL_STATIC_DRAW)
		# enable arrays
		glEnableVertexAttribArray(self.vertIndex)
		glEnableVertexAttribArray(self.texIndex)
		# Position attribute
		glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
		glVertexAttribPointer(self.vertIndex, 3, GL_FLOAT, GL_FALSE, 0,None)
		# TexCoord attribute
		glBindBuffer(GL_ARRAY_BUFFER, self.tcBuffer)				
		glVertexAttribPointer(self.texIndex, 2, GL_FLOAT, GL_FALSE, 0,None)
		
		# unbind VAO
		glBindVertexArray(0)
		glBindBuffer(GL_ARRAY_BUFFER,0)
	def render(self):		   
		# enable texture	
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D,self.tex_id)
		# use shader
		# set proj matrix
		glUniformMatrix4fv(glGetUniformLocation(self.program, 'uPMatrix'),1,GL_FALSE, pMatrix)		   
		# set modelview matrix
		glUniformMatrix4fv(glGetUniformLocation(self.program, 'uMVMatrix'),1, GL_FALSE, mvMatrix)
		glUseProgram(self.program)
		glUniform1f(glGetUniformLocation(self.program,"a"),self.a)
		glUniform1f(glGetUniformLocation(self.program,"b"),self.b)
		glUniform1f(glGetUniformLocation(self.program,"c"),self.c)
		glUniform1f(glGetUniformLocation(self.program,"scale"),self.scale)
		theta = self.r*PI/180.0
		glUniform1f(glGetUniformLocation(self.program,"theta"), theta)
		# bind VAO
		glBindVertexArray(self.vao)
		glEnable(GL_DEPTH_TEST)
		# draw
		glDrawArrays(GL_TRIANGLES, 0, 36)
		# unbind VAO
		glBindVertexArray(0)
tars_dic={}
def build():
	global tars_dic
	global tex_Cu
	global tex_tar
	tex_box=glutils.loadTexture("box.png")
	tex_wood=glutils.loadTexture("wood.png")
	tex_grass=glutils.loadTexture("grass.png")
	tex_metal=glutils.loadTexture("metal.png")
	tex_sky=glutils.loadTexture("sky.png")
	tex_Cu=glutils.loadTexture("Cu.png")
	tex_tar=glutils.loadTexture("target.png")
	for i in range(-21,22):
		for j in range(-21,22):
			if i==-21 or i==21 or j==-21 or j==21:
				height_dic[(i,j)]=100
			else:
				height_dic[(i,j)]=0
	for i in range(-10,11):
		for j in range(-10,11):
			if (i+j)%2==0:
				t=thing(cube_vertices,cube_tex_ma,tex_wood,i,j,-1,1,0)
			else:
				t=thing(cube_vertices,cube_tex_ma,tex_grass,i,j,-1,1,0)
			things.append(t)
			height_dic[(i,j)]+=1
	level=[[(-3,3),(-2,3),(-1,3),(0,3),(1,3),(2,3),(3,3),(-3,4),(-2,4),(-1,4),(0,4),(1,4),(2,4),(3,4),(-2,5),(-1,5),(0,5),(1,5),(2,5),(-1,6),(0,6),(1,6),(0,7)],
	[(-2,5),(-1,5),(0,5),(1,5),(2,5),(-1,6),(0,6),(1,6),(0,7)],[(-1,6),(0,6),(1,6),(0,7)],[(0,7)]]
	for L in range(4):
		for each in level[L]:
			t=thing(cube_vertices,cube_tex_ma,tex_box,each[0],each[1],L,1,0)
			height_dic[(each[0],each[1])]+=1
			things.append(t)
	tars=[(-2,3),(-4,5),(5,7),(0,0),(4,5),(-7,-8),(-4,-2)]
	for each in tars:
		t=thing(cube_vertices,cube_tex_ma,tex_tar,each[0],each[1],5,1,0)
		t.is_tar=1
		tars_dic[(each[0],each[1])]=t
		things.append(t)
	t2=thing(surface_vertices,surface_tex_ma,tex_sky,1,0,5,40,0)
	things.append(t2)
	t3=thing(surface_vertices,surface_tex_ma,tex_metal,1,0,-2,40,0)
	things.append(t3)	
	t4=thing(surface_vertices2,surface_tex_ma,tex_sky,0,20,0,1,0)
	things.append(t4)
	t5=thing(surface_vertices2,surface_tex_ma,tex_sky,0,-20,0,1,0)
	things.append(t5)	
	t6=thing(surface_vertices3,surface_tex_ma,tex_sky,20,0,0,1,0)
	things.append(t6)
	t7=thing(surface_vertices3,surface_tex_ma,tex_sky,-20,0,0,1,0)
	things.append(t7)
def draw_things():
	for each in things:
		each.render()
def on_key(window, key, scancode, action, mods):
	global squat
	if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
		glfw.set_window_should_close(window,1)
	elif key >0  and key<1024 :
		if action == glfw.PRESS:
			keys[key]=True
		elif (action == glfw.RELEASE):
			keys[key]=False
mouses={}
mouses[0]=False
mouses[1]=False
mouses[2]=False
#squat=0
def on_mouse_button(window,button,action,mods):
	global mouses
	#global squat
	if action== glfw.PRESS:
		mouses[button]=True
	elif action==glfw.RELEASE:
		mouses[button]=False
#def mouse_button_callback():
cam_up_speed=0
def do_movement():
	global tex_Cu
	global cameraPos
	global cameraFront
	global cameraUp
	global deltaTime
	global cam_up_speed
	for each in activate_things:
		each.a+=deltaTime*each.speed[0]
		each.b+=deltaTime*each.speed[1]
		each.c+=deltaTime*each.speed[2]
		if each.c>=5:
			if (round(each.a),round(each.b) ) in tars_dic.keys():
				things.remove(tars_dic[(round(each.a),round(each.b))])
				del tars_dic[(round(each.a),round(each.b))]
		if each.a<-20 or each.a>20 or each.b<-20 or each.b>20 or each.c<-1 or each.c>6 or height_dic[ (round(each.a),round(each.b))]>each.c+1:
			things.remove(each)
			activate_things.remove(each)
	cameraSpeed=5.0*deltaTime
	cameraFront_without_z=cameraFront.copy()
	cameraFront_without_z[2]=0
	pos_keep=cameraPos.copy()
	H2=height_dic[ (round(pos_keep[0]),round(pos_keep[1]))]
	if H2<cameraPos[2] or cam_up_speed!=0:
		cameraPos[2]+=cam_up_speed* deltaTime
		cam_up_speed-=10*deltaTime
	if cameraPos[2]<H2:
		cameraPos[2]=H2
		cam_up_speed=0
	if keys[glfw.KEY_F]:
		T=thing(cube_vertices,cube_tex_ma,tex_Cu,cameraPos[0]+cameraFront[0],cameraPos[1]+cameraFront[1],cameraPos[2]+cameraFront[2],0.2,0)
		T.speed=(5*cameraFront[0],5*cameraFront[1],5*cameraFront[2])
		things.append(T)
		activate_things.append(T)		
	if (keys[glfw.KEY_W]):
		cameraPos += cameraSpeed * cameraFront_without_z
	if (keys[glfw.KEY_S]):
		cameraPos -= cameraSpeed * cameraFront_without_z
	if (keys[glfw.KEY_A]):
		# normalize up vector
		norm = np.linalg.norm(cameraFront_without_z)
		cameraFront_without_z /= norm
		norm = np.linalg.norm(cameraUp)
		cameraUp /= norm
		# Side = forward x up 
		side = np.cross(cameraFront_without_z, cameraUp)
		cameraPos -= cameraSpeed * side
	if (keys[glfw.KEY_D]):
		# normalize up vector
		norm = np.linalg.norm(cameraFront_without_z)
		cameraFront_without_z /= norm
		norm = np.linalg.norm(cameraUp)
		cameraUp /= norm
		# Side = forward x up 
		side = np.cross(cameraFront_without_z, cameraUp)
		cameraPos += cameraSpeed * side
	if (keys[glfw.KEY_SPACE]):
		if cam_up_speed==0:
			cam_up_speed=5
	H=height_dic[ (round(cameraPos[0]),round(cameraPos[1]))]
	if cameraPos[2]<H:
		cameraPos[0]=pos_keep[0]
		cameraPos[1]=pos_keep[1]
ss_x,ss_y=pyautogui.size()
def mouse_callback(window, xpos, ypos):
	global cameraFront
	global firstMouse
	global lastX
	global lastY
	global yaw
	global pitch
	if (firstMouse==True):
		lastX = xpos
		lastY = ypos
		firstMouse = False
	xoffset=xpos-lastX
	yoffset=lastY-ypos
	if abs(xpos-0)<screen_size[0]/8 or abs(xpos-screen_size[0])<screen_size[0]/8 or abs(ypos-0)<screen_size[1]/8 or abs(ypos-screen_size[1])<screen_size[1]/8:
		pyautogui.moveTo(ss_x/2,ss_y/2)
	lastX=xpos
	lastY=ypos
	if abs(xoffset)+abs(yoffset)>100:
		return
	sensitivity = 0.05
	xoffset *= sensitivity
	yoffset *= sensitivity
	yaw+=xoffset
	pitch+=yoffset
	if pitch>40:
		pitch=40
	if pitch<-40:
		pitch=-40
	cameraFront=[math.sin(math.radians(yaw))*math.cos(math.radians(pitch)),math.cos(math.radians(yaw))*math.cos(math.radians(pitch)),math.sin(math.radians(pitch))]	
	norm = np.linalg.norm(cameraFront)
	cameraFront /= norm
if __name__ == '__main__':
	global cameraPos
	global cameraFront
	global cameraUp
	global deltaTime
	global firstMouse
	global lastX
	global lastY
	global yaw
	global pitch
	
	keys=np.zeros(1024)
	deltaTime=0.0
	lastFrame=0.0
	firstMouse = True
	lastX= 400
	lastY=300
	yaw=-90.0
	pitch=0.0	
	camera = glutils.Camera([0.0, 0.0, 5.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0])
	cameraPos  =np.array([0,0,1], np.float32)
	cameraFront=np.array([1,0.0], np.float32)
	cameraUp   =np.array([0,0,1], np.float32)
	# Initialize the library
	if not glfw.init():
		sys.exit()
	# Create a windowed mode window and its OpenGL context
	window = glfw.create_window(screen_size[0],screen_size[1], "draw Cube ", None, None)
	if not window:
		glfw.terminate()
		sys.exit()
	# Make the window's context current
	glfw.make_context_current(window)
	#glfw.set_input_mode(window, glfw.cursor, glfw.GLFW_CURSOR_HIDDEN)
	glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
	# Install a key handler
	glfw.set_key_callback(window, on_key)
	glfw.set_mouse_button_callback(window,on_mouse_button)
	# set window  mouse callbacks
	glfw.set_cursor_pos_callback(window,  mouse_callback)
	build()
	glEnable(GL_MULTISAMPLE)
	glEnable(GL_DEPTH_TEST)
	glShadeModel(GL_SMOOTH)  # most obj files expect to be smooth-shaded	
	while not glfw.window_should_close(window):
		currentFrame = glfw.get_time()
		deltaTime = currentFrame - lastFrame		   
		lastFrame = currentFrame
		glfw.poll_events()
		do_movement()
		# Render here
		width, height = glfw.get_framebuffer_size(window)
		ratio = width / float(height)
		gl.glViewport(0, 0, width, height)
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glMatrixMode(gl.GL_PROJECTION)
		gl.glLoadIdentity()
		gl.glFrustum(-ratio, ratio, -1, 1, 1, 2)
		gl.glMatrixMode(gl.GL_MODELVIEW)
		gl.glLoadIdentity()
		gl.glClearColor(0.0,0.0,0.0,0.0)
		camera.eye=[5*math.sin(glfw.get_time()), 0 , 5* math.cos(glfw.get_time()) ]
		# modelview matrix
		cameraPos_=cameraPos+(0,0,-0.25*keys[glfw.KEY_X])
		mvMatrix = glutils.lookAt(cameraPos_, cameraPos_+cameraFront, cameraUp)
		pMatrix = glutils.perspective(45, ratio, 0.1, 100.0)					  
		draw_things()
		# Swap front and back buffers
		glfw.swap_buffers(window)		   
		glfw.poll_events()
	glfw.terminate()
