import tkinter as tk
from tkinter import ttk
import numpy as np
import skfuzzy as fuzz

# ========================
# Domínios dos universos fuzzy
# ========================
x_febre    = np.arange(0, 5.1, 0.1)
x_tosse    = np.arange(0, 5.1, 0.1)
x_garganta = np.arange(0, 5.1, 0.1)
x_peito    = np.arange(0, 5.1, 0.1)
x_ar       = np.arange(0, 5.1, 0.1)
x_coriza   = np.arange(0, 5.1, 0.1)
x_espirros = np.arange(0, 5.1, 0.1)
x_fadiga   = np.arange(0, 5.1, 0.1)
x_cabeca   = np.arange(0, 5.1, 0.1)

# ========================
# Funções Fuzzy
# ========================
def get_memberships(febre, tosse, garganta, peito, ar=0, coriza=0, espirros=0, fadiga=0, cabeca=0):
    # Febre
    febre_vals = {
        'gripe':       fuzz.interp_membership(x_febre, fuzz.trapmf(x_febre, [0, 2.5, 3, 5.5]), febre),
        'resfriado':  fuzz.interp_membership(x_febre, fuzz.trapmf(x_febre, [0, 0, 2, 4]), febre),
        'bronquite':  fuzz.interp_membership(x_febre, fuzz.trapmf(x_febre, [-0.5, 1,3, 5]), febre),
        'pneumonia':  fuzz.interp_membership(x_febre, fuzz.trapmf(x_ar, [2.5, 4, 5,5]), febre),
        'asma':        fuzz.interp_membership(x_febre, fuzz.trimf(x_febre, [0, 0, 3]), febre),
        'sinusite':   fuzz.interp_membership(x_febre, fuzz.trimf(x_febre, [0, 0, 3]), febre),
        'rinite':     fuzz.interp_membership(x_febre, fuzz.trimf(x_febre, [0, 0, 1]), febre),
    }

    # Tosse
    tosse_vals = {
        'gripe':       fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [-0.5, 2, 4, 5.5]), tosse),
        'resfriado':  fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [-1, 1, 2, 4]), tosse),
        'bronquite':  fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [-1, 3, 5, 5]), tosse),
        'pneumonia':  fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [0, 3, 5, 5]), tosse),
        'asma':        fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [0, 3, 5, 5]), tosse),
        'sinusite':   fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [0, 0, 1, 3]), tosse),
        'rinite':     fuzz.interp_membership(x_tosse, fuzz.trapmf(x_tosse, [-0.5, 1, 2, 5]), tosse),
    }

    # Dor de Garganta
    garganta_vals = {
        'gripe':       fuzz.interp_membership(x_garganta, fuzz.trapmf(x_garganta, [0, 2, 3, 5]), garganta),
        'resfriado':  fuzz.interp_membership(x_garganta, fuzz.trapmf(x_garganta, [-1, 1, 2, 4]), garganta),
        'bronquite':  fuzz.interp_membership(x_garganta, fuzz.trapmf(x_garganta, [0, 2, 3, 4]), garganta),
        'pneumonia':  fuzz.interp_membership(x_garganta, fuzz.trapmf(x_garganta, [0, 0, 2, 4]), garganta),
        'asma':        fuzz.interp_membership(x_garganta, fuzz.trimf(x_garganta, [0, 0, 0]), garganta),
        'sinusite':   fuzz.interp_membership(x_garganta, fuzz.trimf(x_garganta, [0, 0, 1.5]), garganta),
        'rinite':     fuzz.interp_membership(x_garganta, fuzz.trimf(x_garganta, [0, 0, 2]), garganta),
    }

    # Dor no Peito
    peito_vals = {
        'gripe':       fuzz.interp_membership(x_peito, fuzz.trimf(x_peito, [0, 0, 3]), peito),
        'resfriado':  fuzz.interp_membership(x_peito, fuzz.trimf(x_peito, [0, 0, 0]), peito),
        'bronquite':  fuzz.interp_membership(x_peito, fuzz.trapmf(x_peito, [0.5, 2, 5, 5]), peito),
        'pneumonia':  fuzz.interp_membership(x_peito, fuzz.trapmf(x_peito, [1.5, 4, 5, 5]), peito),
        'asma':        fuzz.interp_membership(x_peito, fuzz.trapmf(x_peito, [0, 3, 4, 5.5]), peito),
        'sinusite':   fuzz.interp_membership(x_peito, fuzz.trimf(x_peito, [0, 0, 0]), peito),
        'rinite':     fuzz.interp_membership(x_peito, fuzz.trimf(x_peito, [0, 0, 0]), peito),
    }

    # Falta de Ar
    ar_vals = {
        'gripe':       fuzz.interp_membership(x_ar, fuzz.trimf(x_ar, [0, 0, 2]), ar),
        'resfriado':  fuzz.interp_membership(x_ar, fuzz.trimf(x_ar, [0, 0, 0]), ar),
        'bronquite':  fuzz.interp_membership(x_ar, fuzz.trapmf(x_ar, [0, 2, 3, 5]), ar),
        'pneumonia':  fuzz.interp_membership(x_ar, fuzz.trapmf(x_ar, [2.5, 4, 5,5]), ar),
        'asma':        fuzz.interp_membership(x_ar, fuzz.trapmf(x_ar, [2.5, 4, 5,5]), ar),
        'sinusite':   fuzz.interp_membership(x_ar, fuzz.trimf(x_ar, [0, 0, 0]), ar),
        'rinite':     fuzz.interp_membership(x_ar, fuzz.trimf(x_ar, [0, 0, 0]), ar),
    }

    # Coriza
    coriza_vals = {
        'gripe':       fuzz.interp_membership(x_coriza, fuzz.trimf(x_coriza, [0, 1.5, 5]), coriza),
        'resfriado':  fuzz.interp_membership(x_coriza, fuzz.trimf(x_coriza, [-1, 3, 5.5]), coriza),
        'bronquite':  fuzz.interp_membership(x_coriza, fuzz.trimf(x_coriza, [0, 0, 2]), coriza),
        'pneumonia':  fuzz.interp_membership(x_coriza, fuzz.trimf(x_coriza, [0, 0, 2]), coriza),
        'asma':        fuzz.interp_membership(x_coriza, fuzz.trimf(x_coriza, [0, 0, 2]), coriza),
        'sinusite':   fuzz.interp_membership(x_coriza, fuzz.trimf(x_coriza, [0, 2, 4]), coriza),
        'rinite':     fuzz.interp_membership(x_coriza, fuzz.trapmf(x_espirros, [1, 3, 5, 5]), coriza),
    }

    # Espirros
    espirros_vals = {
        'gripe':       fuzz.interp_membership(x_espirros, fuzz.trimf(x_espirros, [0, 1, 5]), espirros),
        'resfriado':  fuzz.interp_membership(x_espirros, fuzz.trimf(x_espirros, [-1, 2.5, 4]), espirros),
        'bronquite':  fuzz.interp_membership(x_espirros, fuzz.trimf(x_espirros, [0, 0, 0]), espirros),
        'pneumonia':  fuzz.interp_membership(x_espirros, fuzz.trimf(x_espirros, [0, 0, 0]), espirros),
        'asma':        fuzz.interp_membership(x_espirros, fuzz.trimf(x_espirros, [0, 0, 0]), espirros),
        'sinusite':   fuzz.interp_membership(x_espirros, fuzz.trapmf(x_espirros, [0, 0, 1, 4]), espirros),
        'rinite':     fuzz.interp_membership(x_espirros, fuzz.trapmf(x_espirros, [1, 3,5, 5]), espirros),
    }

    # Fadiga
    fadiga_vals = {
        'gripe':       fuzz.interp_membership(x_fadiga, fuzz.trapmf(x_fadiga, [0, 2, 3, 5.5]), fadiga),
        'resfriado':  fuzz.interp_membership(x_fadiga, fuzz.trapmf(x_fadiga, [0, 1, 2, 3]), fadiga),
        'bronquite':  fuzz.interp_membership(x_fadiga, fuzz.trapmf(x_fadiga, [0, 2, 3, 6]), fadiga),
        'pneumonia':  fuzz.interp_membership(x_fadiga, fuzz.trapmf(x_fadiga, [0, 4, 5, 6]), fadiga),
        'asma':        fuzz.interp_membership(x_fadiga, fuzz.trimf(x_fadiga, [-0.5, 2, 4]), fadiga),
        'sinusite':   fuzz.interp_membership(x_fadiga, fuzz.trimf(x_fadiga, [-0.5, 1, 2]), fadiga),
        'rinite':     fuzz.interp_membership(x_fadiga, fuzz.trimf(x_fadiga, [-0.5, 1, 2]), fadiga),
    }

    # Dor de Cabeça
    dor_cabeca_vals = {
        'gripe':       fuzz.interp_membership(x_cabeca, fuzz.trimf(x_cabeca, [0, 2, 5.5]), cabeca),
        'resfriado':  fuzz.interp_membership(x_cabeca, fuzz.trimf(x_cabeca, [-0.5, 1, 3]), cabeca),
        'bronquite':  fuzz.interp_membership(x_cabeca, fuzz.trimf(x_cabeca, [-0.5, 1, 3]), cabeca),
        'pneumonia':  fuzz.interp_membership(x_cabeca, fuzz.trimf(x_cabeca, [0, 2, 4]), cabeca),
        'asma':        fuzz.interp_membership(x_cabeca, fuzz.trimf(x_cabeca, [-0.5, 1, 3]), cabeca),
        'sinusite':   fuzz.interp_membership(x_cabeca, fuzz.trapmf(x_cabeca, [0, 2, 3, 5]), cabeca),
        'rinite':     fuzz.interp_membership(x_cabeca, fuzz.trimf(x_cabeca, [-1, 2, 4]), cabeca),
    }

    # Agregando os resultados fuzzy para cada doença utilizando o operador "min"
    doencas = {}
    for doenca in febre_vals:
        doencas[doenca] = min(
            febre_vals[doenca],
            tosse_vals[doenca],
            garganta_vals[doenca],
            peito_vals[doenca],
            ar_vals[doenca],
            coriza_vals[doenca],
            espirros_vals[doenca],
            fadiga_vals[doenca],
            dor_cabeca_vals[doenca]
        )
    return doencas

# ========================
# Interface Gráfica
# ========================

root = tk.Tk()
root.title("Triagem de Doenças Respiratórias")

# Configuração de estilo
style = ttk.Style()
style.theme_use('clam')  # Um tema mais moderno

# Emojis para os sintomas
emojis_sintomas = {
    "febre":    ["😊", "🤒", "🥵"],
    "tosse":    ["😊", "😶", " 🗣️"],
    "garganta": ["😊", "😐", " 😫"],
    "peito":    ["😊", "😐", " 😣"],
    "ar":       ["😊", "😮‍💨", " 😨"],
    "coriza":   ["😊", "🤧", " 🤧"],
    "espirros": ["😊", "😥", " 😤"],
    "fadiga":   ["😊", "🥱", " 😫"],
    "cabeca":   ["😊", "🤕", " 🤯"]
}

frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky='nsew')

# Variáveis para os sintomas
febre_var    = tk.DoubleVar()
tosse_var    = tk.DoubleVar()
garganta_var = tk.DoubleVar()
peito_var    = tk.DoubleVar()
ar_var       = tk.DoubleVar()
coriza_var   = tk.DoubleVar()
espirros_var = tk.DoubleVar()
fadiga_var   = tk.DoubleVar()
cabeca_var   = tk.DoubleVar()

valores_labels = {}
emoji_labels = {}

# Slider especial para febre (36-40°C → mapeado para 0-5)
def atualizar_febre_visual(valor):
    valor_c = float(valor)
    valor_fuzzy = (valor_c - 36) * (5 / 4)
    febre_var.set(valor_fuzzy)
    if "febre" in valores_labels:
        valores_labels["febre"].config(text=f"{valor_c:.1f}°C")
    atualizar_emoji("febre", valor_fuzzy)

def atualizar_emoji(nome, valor):
    if nome not in emoji_labels:
        return
    if valor < 1.5:
        emoji_labels[nome].config(text=emojis_sintomas[nome][0])
    elif valor < 3.5:
        emoji_labels[nome].config(text=emojis_sintomas[nome][1])
    else:
        emoji_labels[nome].config(text=emojis_sintomas[nome][2])

ttk.Label(frame, text="Febre (°C):", font=('Arial', 12)).grid(row=0, column=0, sticky='w')
slider_febre = ttk.Scale(frame, from_=36, to=40, orient='horizontal', length=300,
                            command=atualizar_febre_visual)
slider_febre.set(36)
slider_febre.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
label_febre_valor = ttk.Label(frame, text="36.0°C", font=('Arial', 12))
label_febre_valor.grid(row=0, column=2, padx=5, sticky='w')
valores_labels["febre"] = label_febre_valor
slider_febre.set(36)

emoji_labels["febre"] = ttk.Label(frame, text="🌡️", font=('Arial', 16))
emoji_labels["febre"].grid(row=0, column=3, padx=5, sticky='e')

# Funções auxiliares
def atualizar_valor(nome, var):
    if nome not in valores_labels:
        return
    valores_labels[nome].config(text=f"{var.get():.1f}")
    atualizar_emoji(nome, var.get())

sliders = {}
def criar_slider(rotulo, variavel, linha, nome):
    ttk.Label(frame, text=rotulo, font=('Arial', 12)).grid(row=linha, column=0, sticky='w')
    slider = ttk.Scale(frame, from_=0, to=5, orient='horizontal', variable=variavel,
                        length=300, command=lambda val: atualizar_valor(nome, variavel))
    slider.grid(row=linha, column=1, padx=10, pady=5, sticky='ew')
    label_valor = ttk.Label(frame, text="0.0", font=('Arial', 12))
    label_valor.grid(row=linha, column=2, padx=5, sticky='w')
    valores_labels[nome] = label_valor
    sliders[nome] = slider
    emoji_labels[nome] = ttk.Label(frame, text=emojis_sintomas[nome][0], font=('Arial', 16))
    emoji_labels[nome].grid(row=linha, column=3, padx=5, sticky='e')
    return slider

# Criando sliders dos outros sintomas
slider_tosse = criar_slider("Tosse:", tosse_var, 1, "tosse")
slider_garganta = criar_slider("Dor de Garganta:", garganta_var, 2, "garganta")
slider_peito = criar_slider("Dor no Peito:", peito_var, 3, "peito")
slider_ar = criar_slider("Falta de Ar:", ar_var, 4, "ar")
slider_coriza = criar_slider("Coriza:", coriza_var, 5, "coriza")
slider_espirros = criar_slider("Espirros:", espirros_var, 6, "espirros")
slider_fadiga = criar_slider("*Fadiga:", fadiga_var, 7, "fadiga")
slider_cabeca = criar_slider("*Dor de Cabeça:", cabeca_var, 8, "cabeca")

# Diagnóstico
def diagnosticar():
    resultados = get_memberships(
        febre_var.get(),
        tosse_var.get(),
        garganta_var.get(),
        peito_var.get(),
        ar_var.get(),
        coriza_var.get(),
        espirros_var.get(),
        fadiga_var.get(),
        cabeca_var.get()
    )
    resultados_ordenados = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, "Resultado:\n\n")
    for doenca, grau in resultados_ordenados:
        output_text.insert(tk.END, f"{doenca.title()}: {grau:.2f}\n")

# Corrigindo a criação do botão "Diagnosticar"
diagnosticar_button = ttk.Button(frame, text="Diagnosticar", command=diagnosticar, padding=10)
diagnosticar_button.grid(row=9, column=0, columnspan=4, pady=15, sticky='ew')

output_text = tk.Text(frame, height=10, width=50, font=('Arial', 12))
output_text.grid(row=10, column=0, columnspan=4, padx=10, pady=10, sticky='nsew')

# Faz com que a janela principal expanda e redimensione corretamente
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

root.mainloop()
