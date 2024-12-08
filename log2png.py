import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr 
from matplotlib import ticker

def get_scaled_colors(num_max, cmap):
    values = np.linspace(0., 1., num_max, endpoint=True)
    colors = [cmap(value) for value in values]    
    return colors

def shuffle_even_before_odd(lst):
    even_idx_elements = [lst[i] for i in range(len(lst)) if i % 2 == 0]
    odd_idx_elements = [lst[i] for i in range(len(lst)) if i % 2 != 0]
    return even_idx_elements + odd_idx_elements

def log2png(
    logs_names,
    logs_tstep,
    logs_tacc,
    logs_tloss,
    logs_vstep,
    logs_vacc,
    logs_vloss,
    max_epoch,
    linx=False,
    liny=False,
):
    
    # print(logs_names)
    
    # colors1 = plt.cm.hsv(np.linspace(0., 1., len(logs_names), endpoint=True))
    # colors2 = plt.cm.plasma(np.linspace(0., 0.8, len(logs_names), endpoint=True))
    
    colors1 = plt.cm.hsv(np.linspace(0., 1., 8, endpoint=True))
    colors2 = plt.cm.plasma(np.linspace(0., 0.8, 8, endpoint=True))

    colors = np.vstack((colors1, colors2))
    mymap = clr.LinearSegmentedColormap.from_list('custom_map', colors)
    
    colors = get_scaled_colors(
        num_max=len(logs_names), 
        # cmap=plt.get_cmap('tab20')
        cmap=mymap
    )
    # colors = shuffle_even_before_odd(colors)
    alpha=0.7
                    
    fig, axs = plt.subplots(2, 2, figsize=(9, 10), sharex=True, sharey='row')
    suptitle = ''
    tmp_names = [name.split('_')[0] for name in logs_names]
    for unique_name in np.unique(tmp_names):
        suptitle += f'{unique_name} '
    fig.suptitle(suptitle, fontsize=20)

    for it, name in enumerate(logs_names):    
        print(len(logs_tstep), len(logs_tstep[0]), ' hi')
        axs[0,0].scatter(logs_tstep[it], logs_tacc[it], color=colors[it], alpha=alpha, s=2, label=f'{name}')
        axs[0,0].plot(logs_tstep[it], logs_tacc[it], color=colors[it], alpha=alpha, lw=1)
        
        axs[0,1].scatter(logs_vstep[it], logs_vacc[it], color=colors[it], alpha=alpha, s=2)
        axs[0,1].plot(logs_vstep[it], logs_vacc[it], color=colors[it], alpha=alpha, lw=1)
        
        axs[1,0].scatter(logs_tstep[it], logs_tloss[it], color=colors[it], alpha=alpha, s=2)
        axs[1,0].plot(logs_tstep[it], logs_tloss[it], color=colors[it], alpha=alpha, lw=1)
        
        axs[1,1].scatter(logs_vstep[it], logs_vloss[it], color=colors[it], alpha=alpha, s=2)
        axs[1,1].plot(logs_vstep[it], logs_vloss[it], color=colors[it], alpha=alpha, lw=1)
    
    
    for it, ax in enumerate(axs.flatten()):
        ax.set_xlim(logs_tstep[0][0], max_epoch)
        if not linx:
            ax.set_xscale('log')
        if not liny:
            ax.set_yscale('log')

        if it<2:
            y_min = np.min(logs_vacc)
            y_max = np.max(logs_vacc)
            y_offset = 0.022 * (y_max / y_min)
   
            ax.hlines(y=np.max(logs_vacc), xmin=logs_tstep[0][0], xmax=max_epoch, alpha=0.7, color='k', linestyle='-.')
            ax.annotate(xy=(max_epoch/2, np.max(logs_vacc)-0.05), text=f'{np.max(logs_vacc):.2f}')
            ax.set_ylim(0.09, 1.1)
        else:
            ax.set_yscale('log')
            y_min = np.min(logs_vloss)
            y_max = np.max(logs_vloss)
            y_offset = 0.022 * (y_max / y_min)
            y_min *= 1-y_offset
            y_max *= 1+y_offset
            
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('epoch')
            ax.hlines(y=np.min(logs_vloss), xmin=logs_tstep[0][0], xmax=max_epoch, alpha=0.7, color='k', linestyle='-.')
            ax.annotate(xy=(max_epoch/100, np.min(logs_vloss)*(1+0.5*y_offset)), text=f'{np.min(logs_vloss):.2f}')
            
    handles, labels = axs[0, 0].get_legend_handles_labels()
            
    axs[0,0].set_title('Train Accuracy', fontsize=10)
    axs[0,1].set_title('Test Accuracy', fontsize=10)
    axs[1,0].set_title('Train CrossEntropy', fontsize=10)
    axs[1,1].set_title('Test CrossEntropy', fontsize=10)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3) 
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.24), frameon=False, fontsize=11, markerscale=3., ncol=2)
    
    plt.savefig('./tmp.png')
    plt.close("all")
    
    return 0