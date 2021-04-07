__author__ = 'lucabasa'
__version__ = '2.0.0'


import pandas as pd 
import numpy as np 

import gc


def big_wins(data, rank_loc):
    '''
    Takes the Massey Ordinals data and average by team/day
    For each game, merge the team' rank on the day of the game
    If the losing team was in the top 30, it calls it a win against a top team
    If a team beats another one with 15 rank position higher, it calls it an upset
    '''
    df = data.copy()
    
    if rank_loc:
        ranks = pd.read_csv(rank_loc)
        # exclude ranks that are on very different value ranges
        ranks = ranks[~(ranks.SystemName.isin(['AP', 'USA', 'DES', 'LYN', 'ACU', 
                                               'TRX', 'D1A', 'JNG', 'BNT']))].copy()
        mean_ranks = ranks.groupby(['Season', 'TeamID', 'RankingDayNum'], as_index=False).OrdinalRank.mean()

        df = pd.merge(df, mean_ranks.rename(columns={'TeamID': 'WTeamID', 
                                                    'RankingDayNum':'DayNum', 
                                                    'OrdinalRank': 'WRank'}), 
                    on=['Season', 'WTeamID', 'DayNum'], how='left')

        df = pd.merge(df, mean_ranks.rename(columns={'TeamID': 'LTeamID', 
                                                        'RankingDayNum':'DayNum', 
                                                        'OrdinalRank': 'LRank'}), 
                        on=['Season', 'LTeamID', 'DayNum'], how='left')

        df = df.fillna(1000)

        df['Wtop_team'] = 0
        df.loc[df.LRank <= 30, 'Wtop_team'] = 1

        df['Wupset'] = 0
        df.loc[df.WRank - df.LRank > 15, 'Wupset'] = 1
        
        del df['WRank']
        del df['LRank']
    
    df['WOT_win'] = 0
    df.loc[df.NumOT > 0, 'WOT_win'] = 1
    
    df['WAway'] = 0
    df.loc[df.WLoc!='H', 'WAway'] = 1
    
    return df


def perc_OT_win(data):
    df = data[['Season', 'TeamID', 'NumOT', 'OT_win']].copy()
    df['has_OT'] = np.where(df.NumOT > 0, 1, 0)
    
    df = df.groupby(['Season', 'TeamID', 'has_OT'], as_index=False).OT_win.mean()
    df = df[df.has_OT > 0].copy()
    del df['has_OT']
    
    return df.rename(columns={'OT_win': 'OT_win_perc'})


def full_stats(data):
    df = data.copy()
    
    to_select = [col for col in df.columns if col.startswith('W') 
                                             and '_perc' not in col 
                                             and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_W = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_W.columns = df_W.columns.str.replace('W','')
    df_W['N_wins'] = 1
    
    to_select = [col for col in df.columns if col.startswith('L') 
                                             and '_perc' not in col 
                                             and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_L = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_L.columns = df_L.columns.str.replace('L','')
    df_L[[col for col in df.columns if '_diff' in col]] = - df_L[[col for col in df.columns if '_diff' in col]]
    for col in [col for col in df.columns if '_advantage' in col]:
        df_L[col] = df_L[col].map({0:1, 1:0})
    df_L['N_wins'] = 0
    df_L['OT_win'] = 0
    df_L['Away'] = 0
    if 'top_team' in df_W.columns:
        df_L['top_team'] = 0
        df_L['upset'] = 0

    df = pd.concat([df_W, df_L], sort=True)
    
    del df['DayNum']
    
    OT_perc = perc_OT_win(df)
    
    not_use = ['NumOT']
    to_use = [col for col in df.columns if col not in not_use]
    
    means = df[to_use].groupby(['Season','TeamID'], as_index=False).mean()
    
    sums = df[to_use].groupby(['Season','TeamID'], as_index=False).sum()
    sums['FGM_perc'] = sums.FGM / sums.FGA
    sums['FGM2_perc'] = sums.FGM2 / sums.FGA2
    sums['FGM3_perc'] = sums.FGM3 / sums.FGA3
    sums['FT_perc'] = sums.FTM / sums.FTA
    sums['FGM_no_ast_perc'] = sums.FGM_no_ast / sums.FGM
    sums['True_shooting_perc'] = 0.5 * sums['Score'] / (sums['FGA'] + 0.475 * sums['FTA'])
    sums['Opp_True_shooting_perc'] = 0.5 * sums['opp_score'] / (sums['opp_FGA'] + 0.475 * sums['opp_FTA'])
    to_use = ['Season', 'TeamID', 'FGM_perc',
              'FGM2_perc', 'FGM3_perc', 'FT_perc', 
              'FGM_no_ast_perc', 'True_shooting_perc', 'Opp_True_shooting_perc']
    
    sums = sums[to_use].fillna(0)
    
    stats_tot = pd.merge(means, sums, on=['Season', 'TeamID'])
    stats_tot = pd.merge(stats_tot, OT_perc, on=['Season', 'TeamID'], how='left')
    stats_tot['OT_win_perc'] = stats_tot['OT_win_perc'].fillna(0)
  
    return stats_tot


def process_details(data, rank_loc=None):
    '''
    Some extra statistic are calculated for both the winning and the losing team
    It calculates the difference between the two teams in each stat
    '''
    df = data.copy()
    
    df = big_wins(df, rank_loc)
        
    for prefix in ['W', 'L']:
        df[prefix+'FG_perc'] = df[prefix+'FGM'] / df[prefix+'FGA']
        df[prefix+'FGM2'] = df[prefix+'FGM'] - df[prefix+'FGM3']
        df[prefix+'FGA2'] = df[prefix+'FGA'] - df[prefix+'FGA3']
        df[prefix+'FG2_perc'] = df[prefix+'FGM2'] / df[prefix+'FGA2']
        df[prefix+'FG3_perc'] = df[prefix+'FGM3'] / df[prefix+'FGA3']
        df[prefix+'FT_perc'] = df[prefix+'FTM'] / df[prefix+'FTA']
        df[prefix+'Tot_Reb'] = df[prefix+'OR'] + df[prefix+'DR']
        df[prefix+'FGM_no_ast'] = df[prefix+'FGM'] - df[prefix+'Ast']
        df[prefix+'FGM_no_ast_perc'] = df[prefix+'FGM_no_ast'] / df[prefix+'FGM']
        df[prefix+'possessions'] = df[prefix+'FGA'] - df[prefix+'OR'] + df[prefix+'TO'] + 0.475*df[prefix+'FTA']
        df[prefix+'off_rating'] = df[prefix+'Score'] / df[prefix+'possessions'] * 100
        df[prefix+'shtg_opportunity'] = 1 + (df[prefix+'OR'] - df[prefix+'TO']) / df[prefix+'possessions']
        df[prefix+'TO_perposs'] = df[prefix+'TO'] / df[prefix+'possessions']
        df[prefix+'True_shooting_perc'] = 0.5 * df[prefix+'Score'] / (df[prefix+'FGA'] + 0.475 * df[prefix+'FTA'])
        df[prefix+'IE_temp'] = df[prefix+'Score'] + df[prefix+'FTM'] + df[prefix+'FGM'] + \
                                df[prefix+'DR'] + 0.5*df[prefix+'OR'] - df[prefix+'FTA'] - df[prefix+'FGA'] + \
                                df[prefix+'Ast'] + df[prefix+'Stl'] + 0.5*df[prefix+'Blk'] - df[prefix+'PF']

    df['Wdef_rating'] = df['Loff_rating']
    df['Ldef_rating'] = df['Woff_rating']
    df['Wopp_shtg_opportunity'] = df['Lshtg_opportunity']
    df['Lopp_shtg_opportunity'] = df['Wshtg_opportunity']
    df['Wopp_possessions'] = df['Lpossessions']
    df['Lopp_possessions'] = df['Wpossessions']
    df['Wopp_score'] = df['LScore']
    df['Lopp_score'] = df['WScore']
    # These will be needed for the true shooting percentage when we aggregate
    df['Wopp_FTA'] = df['LFTA']
    df['Wopp_FGA'] = df['LFGA']
    df['Lopp_FTA'] = df['WFTA']
    df['Lopp_FGA'] = df['WFGA']

    df['Wimpact'] = df['WIE_temp'] / (df['WIE_temp'] + df['LIE_temp'])
    df['Limpact'] = df['LIE_temp'] / (df['WIE_temp'] + df['LIE_temp'])

    del df['WIE_temp']
    del df['LIE_temp']

    df[[col for col in df.columns if 'perc' in col]] = df[[col for col in df.columns if 'perc' in col]].fillna(0)

    df['WDR_opportunity'] = df['WDR'] / (df['LFGA'] - df['LFGM'])
    df['LDR_opportunity'] = df['LDR'] / (df['WFGA'] - df['WFGM'])
    df['WOR_opportunity'] = df['WOR'] / (df['WFGA'] - df['WFGM'])
    df['LOR_opportunity'] = df['LOR'] / (df['LFGA'] - df['LFGM'])
    
    stats = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 
             'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 
             'PF', 'FGM2', 'FGA2', 'Tot_Reb', 'FGM_no_ast', 
             'DR_opportunity', 'OR_opportunity', 'possessions',
             'off_rating', 'def_rating', 'shtg_opportunity', 
             'TO_perposs', 'impact', 'True_shooting_perc'] # 'Def_effort' 
    
    for col in stats:
        df[col+'_diff'] = df['W'+col] - df['L'+col]
        df[col+'_advantage'] = (df[col+'_diff'] > 0).astype(int)
    
    return df


def add_days(data, info, date=True):
    '''
    Transdorms DayNum into the actual date of the game and viceversa
    '''
    df = data.copy()
    seasons = pd.read_csv(info)
    
    df = pd.merge(df, seasons[['Season', 'DayZero']], on='Season')
    df['DayZero'] = pd.to_datetime(df.DayZero)
    
    if date:
        df['GameDay'] = df.apply(lambda x: x['DayZero'] + pd.offsets.DateOffset(days=x['DayNum']), 1)
    else:
        df['DayNum'] = (df['GameDay'] - df['DayZero']).dt.days
    
    del df['DayZero']
    
    return df


def rolling_stats(data, season_info, window='30d'):
    '''
    For each team in each game, calculates the statistics of the previous 30 days
    The window can be changed
    '''
    df = data.copy()

    df = add_days(df, season_info)

    to_select = [col for col in df.columns if col.startswith('W') 
                                                 and '_perc' not in col 
                                                 and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_W = df[['Season', 'GameDay', 'NumOT', 
               'game_lc', 'half2_lc', 'crunchtime_lc'] + to_select].copy()
    df_W.columns = df_W.columns.str.replace('W','')
    df_W['N_wins'] = 1

    to_select = [col for col in df.columns if col.startswith('L') 
                                             and '_perc' not in col 
                                             and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_L = df[['Season', 'GameDay', 'NumOT', 
               'game_lc', 'half2_lc', 'crunchtime_lc'] + to_select].copy()
    df_L.columns = df_L.columns.str.replace('L','')
    df_L[[col for col in df.columns if '_diff' in col]] = - df_L[[col for col in df.columns if '_diff' in col]]
    for col in [col for col in df.columns if '_advantage' in col]:
        df_L[col] = df_L[col].map({0:1, 1:0})
    df_L['N_wins'] = 0
    df_L['OT_win'] = 0
    df_L['Away'] = 0
    if 'top_team' in df_W.columns:
        df_L['top_team'] = 0
        df_L['upset'] = 0

    df = pd.concat([df_W, df_L], sort=False)

    not_use = ['NumOT', 'Season', 'TeamID']
    to_use = [col for col in df.columns if col not in not_use]

    means = df.groupby(['Season', 'TeamID'])[to_use].rolling(window, on='GameDay', 
                                                           min_periods=1, closed='left').mean()
    means = means.dropna()
    means = means.reset_index()
    del means['level_2']

    sums = df.groupby(['Season', 'TeamID'])[to_use].rolling(window, on='GameDay', 
                                                      min_periods=1, closed='left').sum()
    sums = sums.reset_index()
    del sums['level_2']
    
    sums['FGM_perc'] = sums.FGM / sums.FGA
    sums['FGM2_perc'] = sums.FGM2 / sums.FGA2
    sums['FGM3_perc'] = sums.FGM3 / sums.FGA3
    sums['FT_perc'] = sums.FTM / sums.FTA
    sums['FGM_no_ast_perc'] = sums.FGM_no_ast / sums.FGM
    sums['True_shooting_perc'] = 0.5 * sums['Score'] / (sums['FGA'] + 0.475 * sums['FTA'])
    sums['Opp_True_shooting_perc'] = 0.5 * sums['opp_score'] / (sums['opp_FGA'] + 0.475 * sums['opp_FTA'])
    
    to_use = ['Season', 'TeamID', 'GameDay', 'FGM_perc',
              'FGM2_perc', 'FGM3_perc', 'FT_perc', 
              'FGM_no_ast_perc', 'True_shooting_perc', 'Opp_True_shooting_perc']

    sums = sums[to_use].fillna(0)

    stats_tot = pd.merge(means, sums, on=['Season', 'TeamID', 'GameDay'])

    stats_tot = add_days(stats_tot, season_info, date=False)
    del stats_tot['GameDay']
    
    return stats_tot


def make_scores(data):
    '''
    Uses the made1/made2/made3 events to calculate the score at each event
    '''
    to_keep = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3', 'reb', 'turnover', 'assist', 'steal', 'block']
    df = data[data.EventType.isin(to_keep)].copy()
    to_drop = ['EventPlayerID', 'EventSubType', 'X', 'Y', 'Area']
    df.drop(to_drop, axis=1, inplace=True)
    
    df['tourney'] = np.where(df.DayNum >= 132, 1, 0)
    
    df['points_made'] = 0
    df.loc[df.EventType == 'made1', 'points_made'] = 1
    df.loc[df.EventType == 'made2', 'points_made'] = 2
    df.loc[df.EventType == 'made3', 'points_made'] = 3
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    df['Final_difference'] = df['WFinalScore'] - df['LFinalScore']
    
    df = df.sort_values(by=['DayNum', 'WTeamID', 'ElapsedSeconds'])
    
    df['points'] = df.groupby(['tmp_gameID', 'EventTeamID']).points_made.cumsum() - df.points_made
    
    del df['WCurrentScore']
    del df['LCurrentScore']
    
    df.loc[df.WTeamID == df.EventTeamID, 'WCurrentScore'] = df.points
    df.loc[df.LTeamID == df.EventTeamID, 'LCurrentScore'] = df.points

    df['WCurrentScore'] = df.groupby('tmp_gameID')['WCurrentScore'].fillna(method='ffill').fillna(0)
    df['LCurrentScore'] = df.groupby('tmp_gameID')['LCurrentScore'].fillna(method='ffill').fillna(0)
    
    df['Current_difference'] = df['WCurrentScore'] - df['LCurrentScore']
    
    del df['points']
    del df['points_made']
    del df['tmp_gameID']
    
    return df


def quarter_score(data, men=True):
    '''
    Stores the score at the end of each focus period
    Thus at the end of the game, at the end of the 1st half, or at the 37th minute mark
    '''
    if not men:
        data = data[~((data.DayNum == 80) & (data.WTeamID == 3111) & (data.LTeamID == 3117))]  # fix for one game with odd seconds
    df = data.copy()
    
    df['period'] = 1
    df.loc[df.ElapsedSeconds >= 20 * 60, 'period'] = 2
    df.loc[df.ElapsedSeconds >= 40 * 60, 'period'] = 3
    
    df['crunch'] = 0
    df.loc[(df.ElapsedSeconds > 37 * 60) & (df.ElapsedSeconds <= 40 * 60), 'crunch'] = 1
    
    df['minutes'] = df['ElapsedSeconds'] / 60
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    
    ot = ((df.groupby('tmp_gameID').minutes.max() - 40) / 5).reset_index()
    ot['n_OT'] = np.where(ot.minutes > 0, np.ceil(ot.minutes), 0)    
    half = df[df.period==1].groupby(['tmp_gameID'], as_index=False)[['WCurrentScore', 'LCurrentScore']].max()
    half['Halftime_difference'] = half['WCurrentScore'] - half['LCurrentScore']
    half.drop(['WCurrentScore', 'LCurrentScore'], axis=1, inplace=True)
    crunchtime = df[df.crunch==0].groupby(['tmp_gameID'], as_index=False)[['WCurrentScore', 'LCurrentScore']].max()
    crunchtime['3mins_difference'] = crunchtime['WCurrentScore'] - crunchtime['LCurrentScore']
    crunchtime.drop(['WCurrentScore', 'LCurrentScore'], axis=1, inplace=True)
    
    add_ons = pd.merge(ot[['tmp_gameID', 'n_OT']], half, on='tmp_gameID')
    add_ons = pd.merge(add_ons, crunchtime, on='tmp_gameID')
    
    df = pd.merge(df, add_ons, on='tmp_gameID')
    
    del df['tmp_gameID']
    del df['minutes']
    
    if data.shape[0] != df.shape[0]:
        raise KeyError('Some merge went wrong')
    
    return df


def lead_changes(data):
    '''
    Uses the changes in sign of the current score difference to calculate the number of lead changes in each focus period
    '''
    df = data.copy()
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    
    changes = df.groupby('tmp_gameID').Current_difference.apply(lambda x: len(np.where(np.diff(np.sign(x)))[0])).reset_index()
    changes.rename(columns={'Current_difference': 'game_lc'}, inplace=True)
    changes_2 = df[df.period==2].groupby('tmp_gameID').Current_difference.apply(lambda x: len(np.where(np.diff(np.sign(x)))[0])).reset_index()
    changes_2.rename(columns={'Current_difference': 'half2_lc'}, inplace=True)
    changes_3 = df[df.crunch==1].groupby('tmp_gameID').Current_difference.apply(lambda x: len(np.where(np.diff(np.sign(x)))[0])).reset_index()
    changes_3.rename(columns={'Current_difference': 'crunchtime_lc'}, inplace=True)
    
    add_ons = pd.merge(changes, changes_2, on='tmp_gameID')
    add_ons = pd.merge(add_ons, changes_3, on='tmp_gameID', how='left')
    
    df = pd.merge(df, add_ons, on='tmp_gameID', how='left').fillna(0)
    
    del df['tmp_gameID']
    
    if data.shape[0] != df.shape[0]:
        raise KeyError('Some merge went wrong')
        
    return df


def _scoreinblock(data, text):
    
    df = data.groupby('tmp_gameID', as_index=False)[['WFinalScore', 'LFinalScore', 'WCurrentScore', 'LCurrentScore']].min()
    df[f'Wpoints_made_{text}'] = df['WFinalScore'] - df['WCurrentScore']
    df[f'Lpoints_made_{text}'] = df['LFinalScore'] - df['LCurrentScore']
    
    return df[['tmp_gameID', f'Wpoints_made_{text}', f'Lpoints_made_{text}']]


def _statcount(data, stat, text):
    
    tmp = data.copy()
    tmp['is_stat'] = np.where(tmp.EventType==stat, 1, 0)
    tmp = tmp.groupby(['tmp_gameID', 'EventTeamID'], as_index=False).is_stat.sum()
    
    return tmp.rename(columns={'is_stat': text})


def event_count(data):
    df = data.copy()
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    
    # points made in each block
    half2 = _scoreinblock(df[df.period==2], 'half2')
    crunch = _scoreinblock(df[df.crunch==1], 'crunchtime')
    
    add_ons = pd.merge(half2, crunch, on='tmp_gameID')
    add_ons = pd.merge(add_ons, df[['tmp_gameID', 'WTeamID', 'LTeamID']].drop_duplicates(), on='tmp_gameID')
    
    # stats in each block
    stats = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3', 'reb', 'turnover', 'assist', 'steal', 'block']
    
    period = 'game'    
    for stat in stats:
        name = f'{stat}_{period}'
        to_merge = _statcount(df, stat, name)
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'WTeamID', 
                                                   name: f'W{name}'}), on=['tmp_gameID', 'WTeamID'])
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'LTeamID', 
                                                   name: f'L{name}'}), on=['tmp_gameID', 'LTeamID'])
        gc.collect()
        
    period = 'half2'
    tmp = df[df.period==2]
    for stat in stats:
        name = f'{stat}_{period}'
        to_merge = _statcount(tmp, stat, name)
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'WTeamID', 
                                                   name: f'W{name}'}), on=['tmp_gameID', 'WTeamID'])
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'LTeamID', 
                                                   name: f'L{name}'}), on=['tmp_gameID', 'LTeamID'])
        gc.collect()
        
    period = 'crunchtime'
    tmp = df[df.crunch==1]
    for stat in stats:
        name = f'{stat}_{period}'
        to_merge = _statcount(tmp, stat, name)
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'WTeamID', 
                                                   name: f'W{name}'}), on=['tmp_gameID', 'WTeamID'])
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'LTeamID', 
                                                   name: f'L{name}'}), on=['tmp_gameID', 'LTeamID'])
        gc.collect()
    
    for period in ['game', 'half2', 'crunchtime']:
        # % of scores with assists
        add_ons[f'WAst_perc_{period}'] = (add_ons[f'Wassist_{period}'] / (add_ons[f'Wmade2_{period}'] + add_ons[f'Wmade3_{period}'])).fillna(0)
        add_ons[f'LAst_perc_{period}'] = (add_ons[f'Lassist_{period}'] / (add_ons[f'Lmade2_{period}'] + add_ons[f'Lmade3_{period}'])).fillna(0)
        # % scores
        add_ons[f'WFGM_perc_{period}'] = ((add_ons[f'Wmade2_{period}'] + add_ons[f'Wmade3_{period}'])
                                          / (add_ons[f'Wmade2_{period}'] + add_ons[f'Wmade3_{period}'] + 
                                             add_ons[f'Wmiss2_{period}'] + add_ons[f'Wmiss3_{period}'])).fillna(0)
        add_ons[f'LFGM_perc_{period}'] = ((add_ons[f'Lmade2_{period}'] + add_ons[f'Lmade3_{period}'])
                                          / ((add_ons[f'Lmade2_{period}'] + add_ons[f'Lmade3_{period}']) + 
                                             add_ons[f'Lmiss2_{period}'] + add_ons[f'Lmiss3_{period}'])).fillna(0)
        add_ons[f'WFGM3_perc_{period}'] = (add_ons[f'Wmade3_{period}'] / (add_ons[f'Wmade3_{period}'] + add_ons[f'Wmiss3_{period}'])).fillna(0)
        add_ons[f'LFGM3_perc_{period}'] = (add_ons[f'Lmade3_{period}'] / (add_ons[f'Lmade3_{period}'] + add_ons[f'Lmiss3_{period}'])).fillna(0)
        add_ons[f'WFTM_perc_{period}'] = (add_ons[f'Wmade1_{period}'] / (add_ons[f'Wmade1_{period}'] + add_ons[f'Wmiss1_{period}'])).fillna(0)
        add_ons[f'LFTM_perc_{period}'] = (add_ons[f'Lmade1_{period}'] / (add_ons[f'Lmade1_{period}'] + add_ons[f'Lmiss1_{period}'])).fillna(0)
        
    
    unique_cols = ['Season', 'DayNum', 'tourney', 'tmp_gameID', 'WTeamID', 'LTeamID', 
                   'WFinalScore', 'LFinalScore', 'Final_difference', 'n_OT', 
                   'Halftime_difference', '3mins_difference', 
                   'game_lc', 'half2_lc', 'crunchtime_lc']
    
    to_drop = ['WTeamID', 'LTeamID'] + [col for col in add_ons if 'miss' in col]
    
    df = pd.merge(df[unique_cols].drop_duplicates(), add_ons.drop(to_drop, axis=1), on='tmp_gameID')
    
    del df['tmp_gameID']
    
    return df


def make_competitive(data):
    '''
    Hard-cuts definition of competitive
    '''
    df = data.copy()

    fil = ((df.Final_difference < 4) | (abs(df['3mins_difference']) < 3) | (df.n_OT > 0) | 
         (df.game_lc > 20) | (df.half2_lc > 10) | (df.crunchtime_lc > 2))
    
    df['competitive'] = np.where(fil, 1, 0)
    
    return df


def make_feats(data):
    '''
    Calculates differences, total, and percentages for some statistics
    '''
    df = data.copy()
    
    for col in [col for col in df if 'W' in col and ('_half2' in col or '_crunchtime' in col)]:
        name = col.replace('W', '')
        df[name+'_diff'] = df['W' + name] - df['L' + name]
        
    for col in ['FG_perc', 'FGM_no_ast_perc', 'FT_perc']:
        df[col+'_diff'] = df['W'+col] - df['L'+col]
        
    for col in [col for col in df if 'W' in col and 'TeamID' not in col
            and 'Loc' not in col and '_perc' not in col 
            and '_diff' not in col and 'top_team' not in col 
            and 'upset' not in col and 'OT_win' not in col and 'Away' not in col]:
        name = col.replace('W', '')
        df[name+'_tot'] = df['W' + name] + df['L' + name]
    
    df['Shooting_perc'] = df['FGM_tot'] / df['FGA_tot']
    df['Ast_perc'] = df['Ast_tot'] / df['FGM_tot']
    df['Stl_TO'] = df['Stl_tot'] / df['TO_tot']
    df['OR_perc'] = df['OR_tot'] / df['Tot_Reb_tot']
    df['TO_perposs_tot'] = df['TO_tot'] / df['possessions_tot']
    df['sht_opportunity_tot'] = (df['OR_tot'] - df['TO_tot']) / df['possessions_tot']
    
    df['points_half2_perc'] = df['points_made_half2_tot'] / df['Score_tot']
    df['points_crunchtime_perc'] = df['points_made_crunchtime_tot'] / df['points_made_half2_tot']
    df['reb_half2_perc'] = df['reb_half2_tot'] / df['Tot_Reb_tot']
    df['reb_crunchtime_perc'] = df['reb_crunchtime_tot'] / df['reb_half2_tot']
    df['block_half2_perc'] = (df['block_half2_tot'] / df['Blk_tot']).fillna(0)
    df['block_crunchtime_perc'] = (df['block_crunchtime_tot'] / df['block_half2_tot']).fillna(0)
    df['steal_half2_perc'] = (df['steal_half2_tot'] / df['Stl_tot']).fillna(0)
    df['steal_crunchtime_perc'] = (df['steal_crunchtime_tot'] / df['steal_half2_tot']).fillna(0)
    
    df['block_crunchtime_perc'] = df['block_crunchtime_perc'].replace(np.inf, 0)
    
    for col in [col for col in df if '_diff' in col]:
        df[col] = abs(df[col])
    
    del df['FGM_no_ast_tot']
    del df['FGM_no_ast_diff']
    del df['def_rating_tot']
    del df['def_rating_diff']
    del df['impact_tot']
    del df['Ast_perc_crunchtime_diff']
    df = df.drop([col for col in df if col.startswith('opp_')], axis=1)
    df = df.drop(['made1_half2_tot', 'made2_half2_tot', 'made3_half2_tot'], axis=1)
    df = df.drop(['made1_crunchtime_tot', 'made2_crunchtime_tot', 'made3_crunchtime_tot'], axis=1)
    
    df = df[(df.points_made_crunchtime_tot > 0) & (df.points_made_crunchtime_tot < 100)].copy()
    
    return df
