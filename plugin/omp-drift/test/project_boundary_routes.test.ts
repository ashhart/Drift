import { afterEach, expect, test } from 'bun:test';
import { chmodSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync, mkdirSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { createProjectRouteCheck } from '../../../scripts/omp/project_boundary_routes.mjs';
import { taskRun } from '../../../scripts/omp/task_scope.mjs';
const original = { path:process.env.DRIFT_WORKER_CONFIG, sha:process.env.DRIFT_WORKER_CONFIG_SHA256 };
const roots:string[]=[];
const hash = (raw:Buffer|string) => createHash('sha256').update(raw).digest('hex');
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root,{recursive:true,force:true});
  for (const [key,value] of [['DRIFT_WORKER_CONFIG',original.path],['DRIFT_WORKER_CONFIG_SHA256',original.sha]]) if (value===undefined) delete process.env[key!]; else process.env[key!]=value;
});
function fixture() {
  const root=realpathSync(mkdtempSync(tmpdir()+'/boundary-route-'));roots.push(root);chmodSync(root,0o700);
  const task=root+'/task';mkdirSync(task,{mode:0o700});const artifact=root+'/fixture';writeFileSync(artifact,'public fixture',{mode:0o600});
  const workers=['one','two'].map(id=>({experimental_multi_turn:true,memory_mode:'linked',communication_mode:'text_and_artifacts',session_binding:'fixed',
    identity:{session:id+'-fixed',worker:id,model_id:id,model_sha256:'a'.repeat(64),translator_sha256:'b'.repeat(64)},
    limits:{max_input_bytes:8192,max_output_tokens:128,max_session_tokens:1024,max_turns:2,deadline_ms:1000},context_window:8192,
    expected:{backend:'fixture',nativeStates:['fixture']},command:{executable:'/bin/echo',sha256:hash(readFileSync('/bin/echo')),args:[],cwd:task,env:{}},artifacts:[{path:artifact,sha256:hash(readFileSync(artifact))}]}));
  const owner={version:1,experimental:true,workers};const path=root+'/owner.json';
  const save=()=>{writeFileSync(path,JSON.stringify(owner),{mode:0o600});process.env.DRIFT_WORKER_CONFIG=path;process.env.DRIFT_WORKER_CONFIG_SHA256=hash(readFileSync(path));};save();
  const actors=workers.map((entry,index)=>({role:index?'child':'parent',model:'drift-experimental/'+entry.identity.model_id,worker:entry.identity.worker,session:entry.identity.session}));
  return {root,task,path,artifact,owner,actors,save};
}
test('actual WorkerRoutes preserves the fixed identities and refuses a second OMP harness',()=>{
  const f=fixture(),check=createProjectRouteCheck(f.actors);
  expect(check(f.actors[0],'omp-one',f.task)).toBeUndefined();expect(check(f.actors[0],'omp-one',f.task)).toBeUndefined();
  expect(check(f.actors[1],'omp-two',f.task)).toBeUndefined();
  expect(()=>check(f.actors[0],'other-harness',f.task)).toThrow('PROJECT_BOUNDARY_ROUTE');
  expect(()=>check(f.actors[1],'omp-two',f.task)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
for(const key of ['model','worker','session','role']) test('forged actor '+key+' fails before a route is accepted',()=>{
  const f=fixture();f.actors[0][key]=key==='role'?'child':'forged';
  expect(()=>createProjectRouteCheck(f.actors)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
for(const patch of [{session_binding:undefined},{session_binding:'derived'},{communication_mode:undefined},{communication_mode:'implicit'},
  {experimental_multi_turn:false},{memory_mode:'no-link'}]) test('requires the declared fixed linked channels '+JSON.stringify(patch),()=>{
  const f=fixture();Object.assign(f.owner.workers[0],patch);f.save();
  expect(()=>createProjectRouteCheck(f.actors)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
for(const count of [1,3]) test('route mapping is a bijection with exactly two owner workers '+count,()=>{
  const f=fixture();f.owner.workers=count===1?f.owner.workers.slice(0,1):[...f.owner.workers,{...f.owner.workers[0],identity:{...f.owner.workers[0].identity,worker:'third',model_id:'third',session:'third'}}];f.save();
  expect(()=>createProjectRouteCheck(f.actors)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
test('owner pins and artifact pins are rechecked without exposing payload or paths',()=>{
  const f=fixture(),check=createProjectRouteCheck(f.actors);writeFileSync(f.artifact,'changed PRIVATE_SENTINEL');
  try{check(f.actors[0],'omp',f.task);throw new Error('accepted');}catch(error){expect((error as Error).message).toBe('PROJECT_BOUNDARY_ROUTE');expect(String(error)).not.toContain(f.root);}
});
test('owner config or environment substitution poisons the route validator',()=>{
  const f=fixture(),check=createProjectRouteCheck(f.actors);f.owner.workers[0].identity.session='changed';f.save();
  expect(()=>check(f.actors[0],'omp',f.task)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
test('actor mutation cannot rewrite a captured route',()=>{
  const f=fixture(),check=createProjectRouteCheck(f.actors);f.actors[0].session='changed';
  expect(()=>check(f.actors[0],'omp',f.task)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
for(const fault of ['mode','symlink','size','task-root']) test('owner configuration remains private and bounded '+fault,()=>{
  const f=fixture();
  if(fault==='mode')chmodSync(f.path,0o644);
  if(fault==='symlink'){symlinkSync(f.path,f.root+'/alias.json');process.env.DRIFT_WORKER_CONFIG=f.root+'/alias.json';}
  if(fault==='size'){writeFileSync(f.path,' '.repeat(262145));process.env.DRIFT_WORKER_CONFIG_SHA256=hash(readFileSync(f.path));}
  if(fault==='task-root'){const check=createProjectRouteCheck(f.actors);expect(()=>check(f.actors[0],'omp',f.root)).toThrow('PROJECT_BOUNDARY_ROUTE');}
  else expect(()=>createProjectRouteCheck(f.actors)).toThrow('PROJECT_BOUNDARY_ROUTE');
});
test('model task command cannot read sibling owner config through the OS sandbox',()=>{
  const f=fixture(),check=createProjectRouteCheck(f.actors);check(f.actors[0],'omp',f.task);
  const executable='/bin/cat',config={root:f.task,max_run_ms:3000,max_output_bytes:4096,runs:[{name:'read',executable,sha256:hash(readFileSync(executable)),args:[f.path],runtime_read_roots:['/System','/usr']}]};
  const result=JSON.parse(taskRun(config,'read'));expect(result.exit_code).not.toBe(0);expect(result.stdout).toBe('');
});
