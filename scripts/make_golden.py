"""One-off script: compile the hand-labelled golden set from labels + alternates."""
import pandas as pd, re, html

g = pd.read_csv('golden_sample_raw.csv', index_col=0)

L = {
0:('REP','REP'),1:('DELIVERY_ADDRESS','auto'),2:('OTHER','auto'),3:('SERVICE_COMPLAINT','human'),4:('REFUND','human'),
5:('SERVICE_COMPLAINT','human'),6:('SERVICE_COMPLAINT','human'),7:('OTHER','auto'),8:('OTHER','auto'),9:('DELIVERY_ADDRESS','human'),
10:('ORDER_STATUS','auto'),11:('DELIVERY_ADDRESS','auto'),12:('DAMAGED_OR_WRONG','auto'),13:('ORDER_STATUS','auto'),14:('OTHER','auto'),
15:('ORDER_STATUS','auto'),16:('OTHER','auto'),17:('ORDER_STATUS','human'),18:('ORDER_STATUS','auto'),19:('REVIEW_CONTENT','human'),
20:('ACCOUNT','auto'),21:('DELIVERY_ADDRESS','auto'),22:('ORDER_STATUS','auto'),23:('APP_WEBSITE','human'),24:('PAYMENT','auto'),
25:('REFUND','human'),26:('DAMAGED_OR_WRONG','auto'),27:('DELIVERY_ADDRESS','auto'),28:('RETURN_CANCEL','human'),29:('ORDER_STATUS','auto'),
30:('RETURN_CANCEL','auto'),31:('REFUND','human'),32:('PAYMENT','auto'),33:('DAMAGED_OR_WRONG','human'),34:('REP','REP'),
35:('PAYMENT','human'),36:('DELIVERY_ADDRESS','human'),37:('REFUND','auto'),38:('OTHER','human'),39:('ORDER_STATUS','human'),
40:('OTHER','auto'),41:('OTHER','human'),42:('DELIVERY_ADDRESS','auto'),43:('PAYMENT','human'),44:('SERVICE_COMPLAINT','human'),
45:('DELIVERY_ADDRESS','human'),46:('REFUND','human'),47:('DELIVERY_ADDRESS','auto'),48:('OTHER','auto'),49:('DELIVERY_ADDRESS','auto'),
50:('RETURN_CANCEL','auto'),51:('OTHER','auto'),52:('DELIVERY_ADDRESS','auto'),53:('ORDER_STATUS','auto'),54:('DAMAGED_OR_WRONG','human'),
55:('ORDER_STATUS','auto'),56:('ACCOUNT','human'),57:('REP','REP'),58:('DELIVERY_ADDRESS','auto'),59:('OTHER','auto'),
60:('REVIEW_CONTENT','human'),61:('ACCOUNT','human'),62:('ORDER_STATUS','auto'),63:('DELIVERY_ADDRESS','human'),64:('ACCOUNT','human'),
65:('ORDER_STATUS','auto'),66:('REP','REP'),67:('RETURN_CANCEL','human'),68:('PAYMENT','human'),69:('DAMAGED_OR_WRONG','human'),
70:('DAMAGED_OR_WRONG','human'),71:('ORDER_STATUS','auto'),72:('REFUND','human'),73:('RETURN_CANCEL','auto'),74:('PAYMENT','auto'),
75:('ORDER_STATUS','human'),76:('SERVICE_COMPLAINT','human'),77:('DELIVERY_ADDRESS','auto'),78:('SERVICE_COMPLAINT','human'),79:('REFUND','auto'),
80:('ORDER_STATUS','auto'),81:('ACCOUNT','human'),82:('ACCOUNT','auto'),83:('ORDER_STATUS','auto'),84:('PAYMENT','human'),
85:('ORDER_STATUS','auto'),86:('DAMAGED_OR_WRONG','auto'),87:('OTHER','auto'),88:('OTHER','auto'),89:('OTHER','auto'),
90:('ORDER_STATUS','human'),91:('OTHER','auto'),92:('APP_WEBSITE','auto'),93:('ACCOUNT','human'),94:('REP','REP'),
95:('ORDER_STATUS','auto'),96:('DELIVERY_ADDRESS','auto'),97:('OTHER','auto'),98:('SERVICE_COMPLAINT','human'),99:('ORDER_STATUS','auto'),
100:('ORDER_STATUS','auto'),101:('DAMAGED_OR_WRONG','human'),102:('REFUND','human'),103:('PAYMENT','auto'),104:('ORDER_STATUS','auto'),
105:('REFUND','human'),106:('REP','REP'),107:('ORDER_STATUS','auto'),108:('ACCOUNT','human'),109:('ACCOUNT','auto'),
110:('OTHER','auto'),111:('ORDER_STATUS','auto'),112:('ORDER_STATUS','human'),113:('DELIVERY_ADDRESS','auto'),114:('SERVICE_COMPLAINT','human'),
115:('OTHER','auto'),116:('PAYMENT','human'),117:('DELIVERY_ADDRESS','human'),118:('RETURN_CANCEL','human'),119:('DELIVERY_ADDRESS','auto'),
120:('PAYMENT','human'),121:('DAMAGED_OR_WRONG','human'),122:('RETURN_CANCEL','auto'),123:('ACCOUNT','human'),124:('SERVICE_COMPLAINT','human'),
125:('ORDER_STATUS','human'),126:('DELIVERY_ADDRESS','human'),127:('SERVICE_COMPLAINT','auto'),128:('ORDER_STATUS','human'),129:('ORDER_STATUS','auto'),
130:('RETURN_CANCEL','auto'),131:('REP','REP'),132:('REVIEW_CONTENT','human'),133:('APP_WEBSITE','auto'),134:('PAYMENT','human'),
135:('RETURN_CANCEL','auto'),136:('OTHER','auto'),137:('OTHER','auto'),138:('SERVICE_COMPLAINT','auto'),139:('OTHER','auto'),
140:('DELIVERY_ADDRESS','human'),141:('OTHER','auto'),142:('ACCOUNT','auto'),143:('SERVICE_COMPLAINT','human'),144:('ACCOUNT','human'),
145:('RETURN_CANCEL','human'),146:('DELIVERY_ADDRESS','human'),147:('DAMAGED_OR_WRONG','auto'),148:('ORDER_STATUS','auto'),149:('DELIVERY_ADDRESS','human'),
150:('SERVICE_COMPLAINT','human'),151:('REVIEW_CONTENT','auto'),152:('ACCOUNT','human'),153:('REFUND','auto'),154:('SERVICE_COMPLAINT','human'),
155:('RETURN_CANCEL','auto'),156:('ORDER_STATUS','auto'),157:('ORDER_STATUS','auto'),158:('PAYMENT','human'),159:('ORDER_STATUS','human'),
160:('ORDER_STATUS','human'),161:('REFUND','auto'),162:('OTHER','auto'),163:('ACCOUNT','human'),164:('RETURN_CANCEL','human'),
165:('PAYMENT','human'),166:('SERVICE_COMPLAINT','human'),167:('OTHER','human'),168:('OTHER','auto'),169:('PAYMENT','human'),
170:('APP_WEBSITE','auto'),171:('REP','REP'),172:('DAMAGED_OR_WRONG','auto'),173:('ORDER_STATUS','auto'),174:('DELIVERY_ADDRESS','auto'),
175:('ORDER_STATUS','human'),176:('ORDER_STATUS','auto'),177:('SERVICE_COMPLAINT','human'),178:('APP_WEBSITE','auto'),179:('DELIVERY_ADDRESS','human'),
180:('DELIVERY_ADDRESS','auto'),181:('ACCOUNT','auto'),182:('REP','REP'),183:('OTHER','human'),184:('ORDER_STATUS','auto'),
185:('DAMAGED_OR_WRONG','human'),186:('DELIVERY_ADDRESS','human'),187:('DELIVERY_ADDRESS','human'),188:('OTHER','auto'),189:('ORDER_STATUS','auto'),
190:('DELIVERY_ADDRESS','human'),191:('ORDER_STATUS','human'),192:('ACCOUNT','human'),193:('ACCOUNT','auto'),194:('REP','REP'),
195:('OTHER','auto'),196:('PAYMENT','auto'),197:('ORDER_STATUS','auto'),198:('SERVICE_COMPLAINT','human'),199:('DELIVERY_ADDRESS','human'),
}
lab = pd.DataFrame([(i, *L[i]) for i in range(200)], columns=['ex_id', 'intent', 'route']).set_index('ex_id')
g2 = g.join(lab)
print('non-english replaced:', list(g2[g2['intent'] == 'REP'].index))
kept = g2[g2['intent'] != 'REP'].copy()

ALT = {
 0:('ORDER_STATUS','human'),1:('PAYMENT','auto'),2:('DAMAGED_OR_WRONG','human'),4:('OTHER','auto'),5:('ACCOUNT','human'),
 9:('RETURN_CANCEL','auto'),12:('REFUND','human'),19:('ACCOUNT','human'),23:('ORDER_STATUS','auto'),28:('ACCOUNT','human'),
}

thr = pd.read_parquet('amazon_threads_full.parquet')
rows = thr.merge(thr[['conv_id','depth','text']].query('depth==1').rename(columns={'text':'brand_reply'})[['conv_id','brand_reply']], on='conv_id')
first = rows[rows['depth']==0][['conv_id','text','brand_reply']].copy()

def clean(t):
    t = html.unescape(str(t))
    t = re.sub(r'https?://t\.co/\w+', ' ', t)
    return re.sub(r'\s+',' ',t).strip()

first['text'] = first['text'].map(clean)
first['brand_reply'] = first['brand_reply'].map(clean)
first = first[first['text'].str.len().between(15, 400)].drop_duplicates(subset=['text'])

def is_english(s):
    try:
        s.encode('ascii'); return True
    except Exception:
        return False

first = first[first['text'].map(is_english)]
g0 = pd.read_csv('golden_sample_raw.csv', index_col=0)
pool = first[~first['conv_id'].isin(set(g0['conv_id']))]
COMMON = set('a an the is are was were be been to of in on at for with my me you your it this that i have has had do does did not no yes and or but if so as we they he she them his her its from by about into over after before between during without within please help order delivery item package card account refund return cancel prime'.split())

def eng_score(s):
    ws = re.findall(r"[a-z']+", s.lower())
    return sum(1 for w in ws if w in COMMON) / len(ws) if ws else 0

pool = pool[pool['text'].map(eng_score) > 0.35]
alts = pool.sample(40, random_state=7).reset_index(drop=True)
alt_rows = []
for k, (intent, route) in ALT.items():
    r = alts.iloc[k]
    alt_rows.append({'conv_id': r['conv_id'], 'text': r['text'], 'brand_reply': r['brand_reply'], 'intent': intent, 'route': route})
alt_df = pd.DataFrame(alt_rows)

kept = pd.concat([kept.reset_index(drop=True), alt_df], ignore_index=True)
kept.index.name = 'ex_id'
kept.to_csv('golden_set.csv')
print('FINAL golden set:', len(kept))
print(kept['intent'].value_counts())
print(kept['route'].value_counts())
