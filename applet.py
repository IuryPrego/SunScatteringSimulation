import pygame
import numpy as np

def W(x):
    r = 6378e3
    t = 8179
    delta = 4*np.cos(x)**2*r**2 - 4*(-2*r*t-t**2)
    return -np.cos(x)*r+np.sqrt(delta)/2


def I(x):
    h=6.62607015e-34
    c=299792458
    k=1.380649e-23
    T=5772
    return(2*h*c**2)/(x**5*(np.exp(h*c/(x*k*T))-1))


def Iw(x,w):
    h=6.62607015e-34
    c=299792458
    k=1.380649e-23
    T=5772
    i0 = (2*h*c**2)/(x**5*(np.exp(h*c/(x*k*T))-1))
    N = 2.7e25
    n = 1.000273
    a = 32*np.pi**3/(3*N*x**4)*(n-1)**2
    return i0*np.exp(-a*w)

LEN_MIN = 380
LEN_MAX = 780
LEN_STEP = 5

X = np.array([
    0.000160, 0.000662, 0.002362, 0.007242, 0.019110, 0.043400, 0.084736, 0.140638, 0.204492, 0.264737,
    0.314679, 0.357719, 0.383734, 0.386726, 0.370702, 0.342957, 0.302273, 0.254085, 0.195618, 0.132349,
    0.080507, 0.041072, 0.016172, 0.005132, 0.003816, 0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
    0.236491, 0.304213, 0.376772, 0.451584, 0.529826, 0.616053, 0.705224, 0.793832, 0.878655, 0.951162,
    1.014160, 1.074300, 1.118520, 1.134300, 1.123990, 1.089100, 1.030480, 0.950740, 0.856297, 0.754930,
    0.647467, 0.535110, 0.431567, 0.343690, 0.268329, 0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
    0.040851, 0.028623, 0.019941, 0.013842, 0.009577, 0.006605, 0.004553, 0.003145, 0.002175, 0.001506,
    0.001045, 0.000727, 0.000508, 0.000356, 0.000251, 0.000178, 0.000126, 0.000090, 0.000065, 0.000046,
    0.000033
])

Y = np.array([
    0.000017, 0.000072, 0.000253, 0.000769, 0.002004, 0.004509, 0.008756, 0.014456, 0.021391, 0.029497,
    0.038676, 0.049602, 0.062077, 0.074704, 0.089456, 0.106256, 0.128201, 0.152761, 0.185190, 0.219940,
    0.253589, 0.297665, 0.339133, 0.395379, 0.460777, 0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
    0.875211, 0.923810, 0.961988, 0.982200, 0.991761, 0.999110, 0.997340, 0.982380, 0.955552, 0.915175,
    0.868934, 0.825623, 0.777405, 0.720353, 0.658341, 0.593878, 0.527963, 0.461834, 0.398057, 0.339554,
    0.283493, 0.228254, 0.179828, 0.140211, 0.107633, 0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
    0.015905, 0.011130, 0.007749, 0.005375, 0.003718, 0.002565, 0.001768, 0.001222, 0.000846, 0.000586,
    0.000407, 0.000284, 0.000199, 0.000140, 0.000098, 0.000070, 0.000050, 0.000036, 0.000025, 0.000018,
    0.000013
])

Z = np.array([
     0.000705, 0.002928, 0.010482, 0.032344, 0.086011, 0.197120, 0.389366, 0.656760, 0.972542, 1.282500,
    1.553480, 1.798500, 1.967280, 2.027300, 1.994800, 1.900700, 1.745370, 1.554900, 1.317560, 1.030200,
    0.772125, 0.570060, 0.415254, 0.302356, 0.218502, 0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
    0.030451, 0.020584, 0.013676, 0.007918, 0.003988, 0.001091, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000
])

MATRIX_SRGB_D65 = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252]
])

def interpolate(values, index, offset):
    return values[index] + (values[index + 1] - values[index]) * (offset / LEN_STEP)

def gamma_correct_srgb(value):
    return (1.055 * (value ** (1/2.4)) - 0.055) if value > 0.0031308 else 12.92 * value

def clip(value):
    return max(0, min(1, value))

def calc_rgb(wavelength):
    if wavelength < LEN_MIN or wavelength > LEN_MAX:
        return (0, 0, 0)
    
    index = int((wavelength - LEN_MIN) // LEN_STEP)
    offset = (wavelength - LEN_MIN) % LEN_STEP
    
    x = interpolate(X, index, offset)
    y = interpolate(Y, index, offset)
    z = interpolate(Z, index, offset)
    
    rgb = np.dot(MATRIX_SRGB_D65, [x, y, z])
    rgb = [clip(gamma_correct_srgb(c)) for c in rgb]
    
    return tuple(int(255 * c) for c in rgb)


def cor(teta, atm=False,teta_aux=0):
    
    dlambda = 1999e-10
    plambda0 = 580e-9
    x = np.linspace(plambda0-dlambda,plambda0+dlambda,81)

    rgb = []

    if atm:
        h = I(x) - Iw(x,W(teta))
        min_bar = np.min(h)*81
        v = np.sum(h)
        for c in x:
            rgb.append(np.array(calc_rgb(c*1e9),dtype=np.int64))
        rgb = np.average(rgb,weights=h, axis=0)
        rgb = rgb*(1-min_bar/v) + np.array([255,255,255],dtype=np.int64)*min_bar/v
    else:
        h = Iw(x,W(teta))
        min_bar = np.min(h)*81
        v = np.sum(h)
        x_medio = np.average(x,weights=h)*1e9
        rgb = np.array(calc_rgb(x_medio),dtype=np.int64)*(1-min_bar/v)+ np.array([255,255,255],dtype=np.int64)*min_bar/v

    return rgb * np.cos(teta_aux)


# Inicializa o pygame
pygame.init()

# Configurações da tela
LARGURA, ALTURA = 1200, 700
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Sol e Parede")

imagem = pygame.image.load("night time starry sky background 0611.jpg").convert_alpha()
imagem = pygame.transform.scale(imagem,(LARGURA,ALTURA))
imagem_rect = imagem.get_rect(center=(LARGURA // 2, ALTURA // 2))

# Cores
# Configuração da bola
raio_sol = 20
raio_terra = 2700
teta_sol = -np.pi/2
raio_traj = 500
x_sol = -int(np.cos(teta_sol)* raio_traj)+LARGURA//2
y_sol = int(np.sin(teta_sol)* raio_traj)+ALTURA-100
velocidade_y = 0

def cor_med(teta_base,teta_aux):
    cor_med = np.array(cor(teta_base,True,teta_aux))
    return cor_med


a = 0

clock = pygame.time.Clock()

# Loop principal
rodando = True
while rodando:
    tela.fill((0,0,0))
    imagem.set_alpha(a)
    tela.blit(imagem, imagem_rect)
    dt = clock.tick(30)/1000

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            rodando = False
        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_UP:
                velocidade_y -= .2*dt
            if evento.key == pygame.K_DOWN:
                velocidade_y += .2*dt
        else:
            velocidade_y = 0


    teta_sol += velocidade_y
    x_sol = -int(np.cos(teta_sol)* raio_traj)+LARGURA//2
    y_sol = int(np.sin(teta_sol)* raio_traj)+ALTURA-100
    teta_aux = np.mod(teta_sol-3*np.pi/2,2*np.pi)

    # Desenha a parede
    terra_posi = (LARGURA//2, ALTURA+raio_terra-160)
    
    if teta_aux <= np.pi/2 or teta_aux >= 3*np.pi/2:
        pygame.draw.circle(tela, cor(np.deg2rad(0),True,teta_aux), terra_posi, raio_terra+300)
        pygame.draw.circle(tela, cor(np.deg2rad(20),True,teta_aux), terra_posi, raio_terra+200)
        pygame.draw.circle(tela, cor(np.deg2rad(40),True,teta_aux), terra_posi, raio_terra+100)
        pygame.draw.circle(tela, cor(np.deg2rad(60),True,teta_aux), terra_posi, raio_terra+65)
        pygame.draw.circle(tela, cor(np.deg2rad(80),True,teta_aux), terra_posi, raio_terra+45)
        pygame.draw.circle(tela, cor(teta_aux), terra_posi, raio_terra+13)
        a = 128*(1-np.cos(teta_aux))
    else:
        a = 128
    
    # Desenha a bola
    pygame.draw.circle(tela, (255,255,255), (x_sol, y_sol), raio_sol)
    pygame.draw.circle(tela, (135,165,237), terra_posi, raio_terra)
    pygame.display.flip()

pygame.quit()