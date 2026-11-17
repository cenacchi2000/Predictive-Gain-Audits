"""Arithmetic checks of supplied printed values; no reconstruction of missing runs."""
from pathlib import Path
import json,math
base=[.811,.742,.694,.736,.612,.861,.749,.671,.703,.766,.781,.733]
policy=[.890,.802,.694,.766,.612,.905,.801,.735,.758,.817,.829,.779]
bb=[.769,.754,.751,.758,.765,.762,.771,.744]
bp=[.799,.826,.809,.813,.816,.821,.824,.831]
mean=lambda x:sum(x)/len(x)
obj={'dataset_table':{'base':mean(base),'policy':mean(policy),'gap':mean([a-b for a,b in zip(policy,base)])},'backbone_table':{'base':mean(bb),'policy':mean(bp),'gap':mean([a-b for a,b in zip(bp,bb)])},'headline':{'base':.759,'policy':.824,'gap':.065},'forda_instance':{'printed_gain':.388,'implied_gain':math.log((1-.541)/(1-.931))},'utility_80pct_implied_error_prevalence':{'reference':.8*.081+.2*.414,'policy':.8*.064+.2*.527},'original_role_totals':{'train':sum([100,3601,1000,8926,896,40,268,15,267,137,6599,150]),'test':sum([100,1320,6164,7711,3582,40,293,15,173,138,2199,850])}}
p=Path(__file__).resolve().parents[1]/'results'/'original_arithmetic_audit.json';p.write_text(json.dumps(obj,indent=2));print(json.dumps(obj,indent=2))
