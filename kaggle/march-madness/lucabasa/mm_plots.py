#__author__ = 'lucabasa'
#__version__ = '1.0.0'

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_game(data, all_names, year, day, w_team, l_team):
    
    fil = ((data.WTeamID == w_team) & 
           (data.LTeamID == l_team) & 
           (data.Season == year) & 
           (data.DayNum == day))
    
    team1 = all_names[all_names.TeamID == w_team].TeamName.values[0]
    team2 = all_names[all_names.TeamID == l_team].TeamName.values[0]
    
    df = data[fil]
    
    # text for the box
    textstr = '\n'.join((
                f'Final Score: {df.WFinalScore.max()} - {df.LFinalScore.max()}',
                f'Haltime score difference: {int(df.Halftime_difference.min())}',
                f'Crunchtime score difference: {int(df["3mins_difference"].min())}', 
                f'Lead Changes: {int(df.game_lc.min())}', 
                f'Lead Changes in second half: {int(df.half2_lc.min())}', 
                f'Lead Changes in final 3 minutes: {int(df.crunchtime_lc.min())}'))
    if df.competitive.max() > 0:
        textstr += '\nThe game was competitive'
    else:
        textstr += '\nThe game was not competitive'
    
    n_ot = df.n_OT.astype(int).max()
    
    fig, ax = plt.subplots(2,1,figsize=(18,14), facecolor='#f7f7f7')
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f'{year} Season, Day {day}, {team1} - {team2}', fontsize=18)
    
    df.plot(x='ElapsedSeconds', y='WCurrentScore', ax=ax[0], label='Winner score', color='g')
    df.plot(x='ElapsedSeconds', y='LCurrentScore', ax=ax[0], label='Loser score', color='r')
    df.plot(x='ElapsedSeconds', y='Current_difference', ax=ax[1], color='k')

    ax[1].fill_between(df.ElapsedSeconds, df.Current_difference, 0, 
                       where=df.Current_difference>0, interpolate=True,
                       color='g', alpha=0.5)
    ax[1].fill_between(df.ElapsedSeconds, df.Current_difference, 0, 
                       where=df.Current_difference<0, interpolate=True,
                       color='r', alpha=0.5)
    
    ax[1].axhline(0, linestyle='--', color='r')
    ax[1].legend().set_visible(False)
    ax[0].legend(loc='lower right')
    
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    ax[0].text(0.03, 0.97, textstr, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    
    ax[0].annotate(round(df.Halftime_difference.max(), 0),
            xy=(20*60, df[df.period==1][['WCurrentScore', 'LCurrentScore']].max().max()), 
            xycoords='data', xytext=(-25, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05))

    for axes in ax:
        axes.axvline(20*60, linestyle='dotted', color='k')
        axes.set_xlabel('Seconds', fontsize=12)
        axes.set_xlim((df.ElapsedSeconds.min(), df.ElapsedSeconds.max()))
        if n_ot > 0: # dotted line for each OT
            for i in range(n_ot):
                axes.axvline(40*60 + i*5*60, linestyle='dotted', color='k')

    ax[0].set_title('Team Score', fontsize=18)
    ax[1].set_title('Score difference', fontsize=18)
    
    plt.show()
    

def get_game(data, names, game_lc=False, half_lc=False, crunch_lc=False, 
             half_score=False, crunch_score=False, final_score=False, 
             half_smaller=True, crunch_smaller=True, final_smaller=True,
             use_competitive=False, competitive=True, OT=False, plot=True, verbose=False):
    '''
    Function to plot a random game satisfying the desired conditions
    game_lc: returns the game with most lead changes
    half_lc: returns the game with most lead changes in the second half
    crunch_lc: returns the game with most lead changes in the last 3 minuts
    half_score: filter by the score difference at halftime
    crunch_score: filter by the score difference at the last 3 minutes
    final_score: filter by the score difference at the end of the game
    use_competitive: filter by competitiveness
    OT: filter by number of OT
    '''
    
    fil = (data.Season > 1)
    
    if half_score:
        if half_smaller:
            fil = fil & (abs(data.Halftime_difference) <= half_score)
        else:
            fil = fil & (abs(data.Halftime_difference) >= half_score)
    if crunch_score:
        if crunch_smaller:
            fil = fil & (abs(data['3mins_difference']) <= crunch_score)
        else:
            fil = fil & (abs(data['3mins_difference']) >= crunch_score)
    if final_score:
        if final_smaller:
            fil = fil & (abs(data.Final_difference) <= final_score)
        else:
            fil = fil & (abs(data.Final_difference) >= final_score)
            
    if use_competitive:
        if competitive:
            fil = fil & (data.competitive == 1)
        else:
            fil = fil & (data.competitive == 0)
    
    if OT:
        fil = fil & (data.n_OT == OT)
        
    df = data[fil]
    
    fil = (df.Season > 1)
        
    if game_lc:
        fil = fil & (df.game_lc == df.game_lc.max())
    elif half_lc:
        fil = fil & (df.half2_lc == df.half2_lc.max())
    elif crunch_lc:
        fil = fil & (df.crunchtime_lc == df.crunchtime_lc.max())
        
    df = df[fil]
    
    if df.shape[0] == 0:
        print('No games with the given characteristics')
        return 0
    elif df.shape[0] > 0:
        cols = ['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WFinalScore', 'LFinalScore', 
                'Final_difference', 'n_OT', 'Halftime_difference', '3mins_difference', 
                'game_lc', 'half2_lc', 'crunchtime_lc', 'competitive']
        final = df[cols].drop_duplicates().sample()
        if verbose:
            print(f'Season: {final.Season.min()}')
            print(f'Day number: {final.DayNum.min()}')
            print(f'Final Score: {final.WFinalScore.max()} - {final.LFinalScore.max()}')
            print(f'Haltime score difference: {final.Halftime_difference.min()}')
            print(f'Crunchtime score difference: {final["3mins_difference"].min()}')
            print(f'Lead Changes: {final.game_lc.min()}')
            print(f'Lead Changes in second half: {final.half2_lc.min()}')
            print(f'Lead Changes in final 3 minutes: {final.crunchtime_lc.min()}')
            if final.competitive.max() > 0:
                print('The game was competitive')
            else:
                print('The game was not competitive')
            
    if plot:
        plot_game(data, names, final.Season.min(), final.DayNum.min(), final.WTeamID.min(), final.LTeamID.min())
        
        
def hardcuts_comp(data, title):
    '''
    Density plots with shaded area
    '''
    fig, ax = plt.subplots(2,2, figsize=(18, 12), facecolor='#f7f7f7')
    fig.subplots_adjust(top=0.92)
    fig.suptitle(title, fontsize=18)

    sns.kdeplot(data.game_lc, ax=ax[0][0], legend=False, color='k', linewidth=2)
    kde_x, kde_y = ax[0][0].lines[0].get_data()
    ax[0][0].fill_between(kde_x, kde_y, where=(kde_x>20), 
                    interpolate=True, color='crimson', alpha=0.7)
    ax[0][0].get_yaxis().set_visible(False)
    ax[0][0].set_title('Lead changes in the game', fontsize=14)
    textstr = '\n'.join((
                f'Mean: {round(data.game_lc.mean(), 2)}',
                f'Std deviation: {round(data.game_lc.std(), 2)}',
                f'Competitive games: {round((data.game_lc > 20).mean() * 100, 2)}%'))
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    ax[0][0].text(0.50, 0.90, textstr, transform=ax[0][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    sns.kdeplot(data.half2_lc, ax=ax[0][1], legend=False, color='k', linewidth=2)
    kde_x, kde_y = ax[0][1].lines[0].get_data()
    ax[0][1].fill_between(kde_x, kde_y, where=(kde_x>10), 
                    interpolate=True, color='crimson', alpha=0.7)
    ax[0][1].get_yaxis().set_visible(False)
    ax[0][1].set_title('Lead changes in the second half', fontsize=14)
    textstr = '\n'.join((
                f'Mean: {round(data.half2_lc.mean(), 2)}',
                f'Std deviation: {round(data.half2_lc.std(), 2)}',
                f'Competitive games: {round((data.half2_lc > 10).mean() * 100, 2)}%'))
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    ax[0][1].text(0.50, 0.90, textstr, transform=ax[0][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    sns.kdeplot(abs(data['3mins_difference']), ax=ax[1][0], legend=False, color='k', linewidth=2)
    kde_x, kde_y = ax[1][0].lines[0].get_data()
    ax[1][0].fill_between(kde_x, kde_y, where=(kde_x<3), 
                    interpolate=True, color='crimson', alpha=0.7)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title('Point difference in the last 3 minutes of the game', fontsize=14)
    textstr = '\n'.join((
                f'Mean: {round(abs(data["3mins_difference"]).mean(), 2)}',
                f'Std deviation: {round(abs(data["3mins_difference"]).std(), 2)}',
                f'Competitive games: {round((abs(data["3mins_difference"]) < 3).mean() * 100, 2)}%'))
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    ax[1][0].text(0.50, 0.90, textstr, transform=ax[1][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    sns.kdeplot(data.crunchtime_lc, ax=ax[1][1], legend=False, color='k', linewidth=2, bw=.3)
    kde_x, kde_y = ax[1][1].lines[0].get_data()
    ax[1][1].fill_between(kde_x, kde_y, where=(kde_x>2), 
                    interpolate=True, color='crimson', alpha=0.7)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_title('Lead changes in the last 3 minutes of the game', fontsize=14)
    textstr = '\n'.join((
                f'Mean: {round(data.crunchtime_lc.mean(), 2)}',
                f'Std deviation: {round(data.crunchtime_lc.std(), 2)}',
                f'Competitive games: {round((data.crunchtime_lc > 2).mean() * 100, 2)}%'))
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    ax[1][1].text(0.50, 0.90, textstr, transform=ax[1][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()