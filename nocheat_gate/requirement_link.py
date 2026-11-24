"""One caller-selected documented rule linked to contract and final gate.

This is an authored rule instantiation, not natural-language intent inference or
independent authority creation. The caller retains the trusted source digest.
"""
import ast,hashlib,json,tomllib
from pathlib import PurePosixPath
from dataclasses import asdict
from pathlib import Path
from .contracts import CONTRACT_PATH,resolve_revisions
from .diff_scope import git_show_file
from .gate import evaluate_nocheat_gate

PROFILE='click-style-unstyle-plain-text-v1'
MODULE='src/click/termui.py'
BRIDGE="from click import style, unstyle\ndef clean(text):\n    return unstyle(style(text, fg='red'))\n"
CONTRACT={'source_roots':['src','.'],'contracts':[{'target':'requirement_api:clean',
    'inputs':{'text':{'type':'str','choices':['a','plain','é']}},'ensures':'result == text'}]}

def sha(data):return hashlib.sha256(data).hexdigest()

def contract_fingerprint(contract):
    """Bind the complete declared contract, before internal AST annotation."""
    return sha(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())

def qualify_contract_policy(repo,base,head,*,path,trusted_sha256,contracts):
    """Check caller-authorized provenance bytes, not the truth of their authority.

    No candidate can approve itself. The caller supplies the retained policy hash;
    each approved binding pins its whole contract and unchanged requirement text.
    """
    def need(condition,reason):
        if not condition:raise ValueError(reason)
    def text(value):return isinstance(value,str) and 0<len(value)<=2048
    def safe_path(value):
        need(text(value) and not any(c in value for c in ('\\','\0',':')),
             'INVALID_REQUIREMENT_PATH')
        need(not PurePosixPath(value).is_absolute() and '..' not in PurePosixPath(value).parts,
             'REQUIREMENT_PATH_OUTSIDE_SOURCE')
    def unique(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,'DUPLICATE_POLICY_KEY');result[key]=value
        return result
    try:
        safe_path(path)
        raw=_source(repo,base,path)
        need(len(raw.encode())<=65536 and sha(raw.encode())==trusted_sha256,'REQUIREMENT_POLICY_DIGEST_MISMATCH')
        need(raw==_source(repo,head,path),'REQUIREMENT_POLICY_CHANGED')
        policy=json.loads(raw,object_pairs_hook=unique)
        need(isinstance(policy,dict) and set(policy)=={'bindings','unresolved'},'INVALID_REQUIREMENT_POLICY')
        entries=policy['bindings'];unresolved=policy['unresolved']
        targets={c['target']:c for c in contracts}
        need(isinstance(entries,list) and 1<=len(entries)<=8,'INVALID_REQUIREMENT_BINDINGS')
        need(isinstance(unresolved,list) and len(unresolved)<=8,'INVALID_UNRESOLVED_CHOICES')
        seen=set();approved=[];excluded=[];sources=[]
        for entry in entries:
            need(isinstance(entry,dict),'INVALID_REQUIREMENT_BINDING')
            target=entry.get('target');role=entry.get('role')
            need(isinstance(target,str) and target in targets and target not in seen,'UNBOUND_OR_DUPLICATE_CONTRACT')
            seen.add(target)
            need(role in ('approved','candidate'),'UNKNOWN_REQUIREMENT_ROLE')
            need(entry.get('contract_sha256')==contract_fingerprint(targets[target]),'REQUIREMENT_CONTRACT_BINDING_MISMATCH')
            if role=='candidate':
                need(set(entry)=={'target','role','contract_sha256','origin'} and text(entry['origin']),
                     'INVALID_CANDIDATE_PROVENANCE')
                excluded.append({'target':target,'origin':entry['origin'],'reason':'CANDIDATE_IS_NOT_REQUIREMENT_AUTHORITY'})
                continue
            need(set(entry)=={'target','role','contract_sha256','source'},'INVALID_APPROVED_BINDING')
            source=entry['source']
            need(isinstance(source,dict) and set(source)=={'path','sha256','statement','version','assumptions'},
                 'INVALID_REQUIREMENT_SOURCE_BINDING')
            safe_path(source['path'])
            need(text(source['statement']) and text(source['version']) and isinstance(source['assumptions'],list)
                 and 1<=len(source['assumptions'])<=8 and all(text(s) for s in source['assumptions']),
                 'REQUIREMENT_SCOPE_UNAVAILABLE')
            original=_source(repo,base,source['path'])
            need(len(original.encode())<=65536 and sha(original.encode())==source['sha256']
                 and original==_source(repo,head,source['path']),'REQUIREMENT_SOURCE_BINDING_MISMATCH')
            need(all(s in original for s in [source['statement'],source['version'],*source['assumptions']]),
                 'REQUIREMENT_STATEMENT_OR_SCOPE_UNAVAILABLE')
            approved.append(target);sources.append({'target':target,**source})
        need(seen==set(targets),'UNCLASSIFIED_CONTRACTS')
        for choice in unresolved:
            need(isinstance(choice,dict) and set(choice)=={'target','question'}
                 and choice['target'] in targets and text(choice['question']),'INVALID_UNRESOLVED_CHOICE')
        return {'status':'QUALIFIED_DECLARED_REQUIREMENTS','policy_path':path,'policy_sha256':trusted_sha256,
                'approved_targets':approved,'approved_sources':sources,'excluded_candidates':excluded,
                'unresolved':unresolved,'authority':'CALLER_DECLARED; byte binding is not independent intent validation'}
    except (ValueError,KeyError,TypeError,OSError,SyntaxError,UnicodeError) as error:
        return {'status':'INCONCLUSIVE','reason':str(error)}
def _source(repo,revision,name):
    text=git_show_file(repo,revision,name)
    if text is None:raise ValueError('REQUIREMENT_SOURCE_UNAVAILABLE:'+name)
    return text

def qualify_requirement(repo,base,head,*,profile,trusted_source_sha256):
    try:
        if profile!=PROFILE:raise ValueError('UNREGISTERED_REQUIREMENT_RULE')
        base,head=resolve_revisions(repo,base,head)
        source=_source(repo,base,MODULE)
        if sha(source.encode())!=trusted_source_sha256:raise ValueError('REQUIREMENT_SOURCE_CHANGED')
        projects=[tomllib.loads(_source(repo,rev,'pyproject.toml'))['project'] for rev in (base,head)]
        if any(p.get('name')!='click' for p in projects) or projects[0].get('version')!=projects[1].get('version'):
            raise ValueError('PACKAGE_VERSION_APPLICABILITY_UNRESOLVED')
        tree=ast.parse(source);docs={}
        for name in ('style','unstyle'):
            fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
            docs[name]={'text':ast.get_docstring(fn),'line':fn.lineno,'signature':ast.dump(fn.args)}
        if not docs['style']['text'].startswith('Styles a text with ANSI styles') or not docs['unstyle']['text'].startswith('Removes ANSI styling information'):
            raise ValueError('DOCUMENTED_RULE_NOT_APPLICABLE')
        head_tree=ast.parse(_source(repo,head,MODULE))
        for name,entry in docs.items():
            fn=next(node for node in head_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
            if ast.get_docstring(fn)!=entry['text'] or ast.dump(fn.args)!=entry['signature']:
                raise ValueError('API_REQUIREMENT_OR_SIGNATURE_CHANGED')
        for revision in (base,head):
            if _source(repo,revision,'requirement_api.py')!=BRIDGE:raise ValueError('REQUIREMENT_BRIDGE_CHANGED')
            if json.loads(_source(repo,revision,CONTRACT_PATH))!=CONTRACT:raise ValueError('UNQUALIFIED_CONTRACT_STRENGTH_OR_SCOPE')
        return {'status':'QUALIFIED_RULE_APPLICATION','profile':profile,'base':base,'head':head,
            'requirement_source':{'path':MODULE,'sha256':trusted_source_sha256,'declarations':docs},
            'contract_sha256':sha(_source(repo,base,CONTRACT_PATH).encode()),
            'package_version':projects[0].get('version'),
            'applicability':{'api':['click.style','click.unstyle'],'inputs':['a','plain','é'],
                             'style_options':{'fg':'red'},'input_contains_ansi':False},
            'derivation':'style adds ANSI styling to plain text; unstyle removes styling; the original plain text remains',
            'attribution':'PROGRAMME_AUTHORED_RULE_AND_CALLER_SELECTED_REQUIREMENT',
            'unresolved':'all other input/options/version applicability is unqualified'}
    except (ValueError,KeyError,TypeError,OSError,SyntaxError,StopIteration) as error:
        return {'status':'INCONCLUSIVE','reason':str(error)}

def evaluate_requirement_case(repo,base,head,*,profile,trusted_source_sha256,native_pythons=None):
    linked=qualify_requirement(repo,base,head,profile=profile,trusted_source_sha256=trusted_source_sha256)
    if linked['status']!='QUALIFIED_RULE_APPLICATION':
        return {'status':'INCONCLUSIVE','requirement_link':linked,'gate':None}
    gate=evaluate_nocheat_gate(Path(repo),linked['base'],linked['head'],
        **({'native_pythons':native_pythons} if native_pythons is not None else {}))
    return {'status':'EXECUTED','requirement_link':linked,'gate':asdict(gate),
            'claim_scope':'declared documented-rule instance; not general intended-behavior discovery'}
