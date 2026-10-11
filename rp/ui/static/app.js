"use strict";

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const uid = () => globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const storageKey = "cradle-roleplay-chat-v2";
const defaults = {name:"自由角色扮演",prompt:"你是一位善于角色扮演的叙事者。根据用户提供的角色、情境和世界信息，保持人物性格和场景连续性。自然地融合对白与动作描写，主动推动故事，但不要替用户决定行动或说话。使用中文回应。",temperature:0.7,max_tokens:192};
const state = {view:"chat",overview:null,settings:null,job:null,history:[],inspectedJob:null,signals:[],signalPage:1,signalTotal:0,preset:"quick",chatBusy:false,pollBusy:false,authenticated:true,events:null,toastTimer:null,character:{...defaults},session:uid(),messages:[]};

function restoreChat() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (saved && Array.isArray(saved.messages) && typeof saved.session === "string") {
      state.messages = saved.messages.filter((m) => ["user","assistant"].includes(m.role) && typeof m.content === "string").slice(-100);
      state.session = saved.session;
      state.character = {...defaults,...saved.character};
    }
  } catch { localStorage.removeItem(storageKey); }
}

function saveChat() {
  try { localStorage.setItem(storageKey, JSON.stringify({session:state.session,messages:state.messages,character:state.character})); }
  catch { toast("浏览器存储已满，本次对话仍可继续。",true); }
}

function toast(message, error=false) {
  clearTimeout(state.toastTimer);
  $("toast").textContent = message;
  $("toast").className = `toast${error ? " toast-error" : ""}`;
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; },error ? 7000 : 4000);
}

function confirmAction(message,title="确认操作") {
  const dialog=$("confirmDialog");
  $("confirmTitle").textContent=title;
  $("confirmMessage").textContent=message;
  dialog.returnValue="";
  return new Promise((resolve)=>{
    dialog.addEventListener("close",()=>resolve(dialog.returnValue==="accept"),{once:true});
    dialog.showModal();
  });
}

function showAuth() {
  state.authenticated = false;
  state.events?.close();
  if (!$("authDialog").open) $("authDialog").showModal();
  $("logoutBtn").hidden = false;
}

async function api(path, options={}) {
  const headers = {"Accept":"application/json",...options.headers};
  if (options.method && options.method !== "GET") {
    headers["Content-Type"] = "application/json";
    headers["X-RP-Request"] = "ui";
  }
  const response = await fetch(path,{...options,headers,credentials:"same-origin"});
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("json") ? await response.json() : {};
  if (response.status === 401) { showAuth(); throw new Error(path==="/api/auth/login" ? "访问令牌未通过验证，请检查后重试。" : "请先连接工作空间。"); }
  if (!response.ok) throw new Error(data.error || data.message || `请求失败 (${response.status})`);
  return data;
}

const post = (path,body={}) => api(path,{method:"POST",body:JSON.stringify(body)});
const patch = (path,body) => api(path,{method:"PATCH",body:JSON.stringify(body)});

async function action(button, task) {
  if (button?.disabled) return;
  if (button) { button.disabled = true; button.setAttribute("aria-busy","true"); }
  try { return await task(); }
  catch (error) { toast(error.message || "无法完成操作，请查看连接状态。",true); }
  finally { if (button) { button.disabled = false; button.removeAttribute("aria-busy"); } renderControls(); }
}

function navigate(view) {
  if (!$("view-"+view)) view="chat";
  state.view = view;
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${view}`; el.classList.toggle("active",!el.hidden); });
  document.querySelectorAll("[data-view]").forEach((el) => { el.classList.toggle("active",el.dataset.view===view); el.setAttribute("aria-current",el.dataset.view===view ? "page" : "false"); });
  $("pageName").textContent = ({chat:"对话",training:"训练",learning:"学习记录",models:"模型",settings:"设置"})[view];
  if (location.hash !== `#${view}`) history.replaceState(null,"",`#${view}`);
  if (view==="learning") loadSignals();
  if (view==="training") loadHistory();
  if (view==="models") renderModels();
}

function isRunning(job=state.job) { return !!job && ["running","queued","preparing","stopping"].includes(job.state); }
function serverReady() { return !!state.overview?.server?.ready; }
function fmt(value,digits=1) { return Number.isFinite(Number(value)) && value!==null && value!==undefined ? Number(value).toFixed(digits) : "—"; }
function fmtTime(seconds) { if (!Number.isFinite(seconds)||seconds<0) return "—"; return seconds<60 ? `${Math.floor(seconds)} 秒` : `${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,"0")}`; }
function signalKind(kind) { return ({good:"喜欢",bad:"不喜欢",fix:"修正",edit:"编辑",swipe:"重新生成"})[kind] || kind; }
function jobKind(kind) { return ({sleep:"聊天学习",narration_train:"叙事训练",narration_eval:"模型对比"})[kind] || kind || "训练"; }
function jobState(value) { return ({running:"运行中",queued:"排队中",preparing:"准备中",stopping:"保存中",done:"已完成",completed:"已完成",error:"失败",failed:"失败",stopped:"已停止",interrupted:"已中断",paused:"已暂停",cancelled:"已取消",idle:"待开始"})[value] || value || "待开始"; }

function renderControls() {
  const server=state.overview?.server || {};
  const pausedAuto=!!state.overview?.auto_training?.running;
  $("sendBtn").disabled = state.chatBusy || !serverReady();
  $("newChatBtn").disabled = state.chatBusy;
  $("saveCharacterBtn").disabled = state.chatBusy;
  $("clearChatBtn").disabled = state.chatBusy;
  $("serverStartBtn").disabled = !!server.running || !!server.starting;
  $("serverStartBtn").hidden = !!server.running || !!server.starting;
  $("serverStopBtn").hidden = !server.running && !server.starting;
  $("serverStopBtn").disabled = state.chatBusy || (!!server.external && !server.managed);
  $("serverStopBtn").title=server.external && !server.managed ? "这个服务由其他进程启动，请在原启动环境中停止。" : "停止当前受管理的对话服务";
  $("learnNowBtn").disabled = isRunning() || pausedAuto || !state.overview?.sleep?.ready || state.chatBusy;
  const trainingType=$("trainingType").value;
  $("startTrainingBtn").disabled = isRunning() || pausedAuto || state.chatBusy || !state.overview || (trainingType==="sleep" && !state.overview.sleep?.ready);
  $("evalNarrBtn").disabled = isRunning() || pausedAuto || state.chatBusy || !state.overview;
  const displayed=state.inspectedJob || state.job;
  $("stopJobBtn").disabled = !isRunning(displayed) || displayed?.id!==state.job?.id || displayed?.state==="stopping";
  $("resumeJobBtn").disabled = isRunning() || pausedAuto || !displayed?.checkpoint || !["stopped","interrupted","paused","error","failed","cancelled"].includes(displayed?.state);
  $("downloadLogBtn").disabled = !displayed?.log_tail?.length;
}

function renderOverview(data) {
  const previousReady=serverReady();
  const previousStarting=state.overview?.server?.starting;
  state.overview=data;
  const server=data.server || {};
  const hardware=data.hardware || {};
  document.querySelector(".chat-aside").classList.toggle("needs-server",!server.ready);
  const stats=data.store || {};
  if (hardware.threads) $("trainThreads").max=String(Math.min(64,hardware.threads));
  if (data.settings) { state.settings=data.settings;$("autoLearnToggle").checked=!!data.settings.enabled && !!data.settings.auto_sleep && data.settings.mode!=="collect_only"; }
  const selectedData=$("trainingData").value;
  const hasScale=!!$("trainingData").querySelector("[value=scale]");
  if (data.scale_available && !hasScale) $("trainingData").insertAdjacentHTML("beforeend","<option value=scale>本地扩充数据 · Scale</option>");
  if (!data.scale_available && hasScale) $("trainingData").querySelector("[value=scale]").remove();
  if (selectedData==="scale" && data.scale_available) $("trainingData").value="scale";
  const newSignals=Number(stats.new_signals_since_sleep || 0);
  const threshold=state.settings?.min_signals || 10;
  $("reviewCount").textContent=String(newSignals);
  $("newSignals").textContent=String(newSignals);
  $("signalFill").style.width=`${Math.min(100,newSignals/threshold*100)}%`;
  $("generationCount").textContent=String(stats.generations || 0);
  $("cpuUsage").textContent=hardware.cpu_percent!=null ? `${fmt(hardware.cpu_percent,0)}%` : "—";
  $("ramAvailable").textContent=hardware.ram_available_gb!=null ? `${fmt(hardware.ram_available_gb)} GB` : "—";
  $("diskFree").textContent=hardware.disk_free_gb!=null ? `${fmt(hardware.disk_free_gb,0)} GB` : "—";
  $("hardwareBadge").innerHTML=icon("cpu")+`CPU${hardware.threads ? ` · ${esc(hardware.threads)} threads` : ""}`;
  $("chatAdapterLabel").textContent=server.ready ? `已连接 · ${server.adapter || data.adapter || "base"}` : server.starting ? "模型加载中 · 首次启动需要下载" : "对话服务尚未就绪";
  $("dataDirectory").textContent=data.data_dir || "—";
  $("serverEndpoint").textContent=`${location.origin}/v1`;
  $("serverModel").textContent=server.model_id || data.sillytavern?.model_id || "—";
  $("serverBadge").textContent=server.ready ? "已就绪" : server.starting ? "加载中" : server.error ? "启动失败" : "未启动";
  $("serverBadge").className=`badge${server.ready ? " badge-success" : server.error ? " badge-error" : ""}`;
  $("serverDescription").textContent=server.error || (server.ready ? server.external && !server.managed ? "正在使用外部启动的模型服务。可正常聊天；请在原启动环境停止服务。" : "模型已就绪，可以在此聊天或连接 SillyTavern。" : server.starting ? "正在加载模型。首次需要下载权重，请保留此窗口；你可以先调整角色和学习设置。" : "启动后，模型会加载到本机内存。第一次需要下载基础模型。");
  $("statusPill").className=`status-pill${server.ready ? " connected" : isRunning(data.job) ? " running" : ""}`;
  $("statusPill").innerHTML=`<span></span>${server.ready ? "对话已就绪" : server.starting ? "模型加载中" : isRunning(data.job) ? "正在训练" : "工作空间已连接"}`;
  $("trainingAvailability").textContent=$("trainingType").value==="sleep" ? (data.sleep?.ready ? `${data.sleep.preferred_replies || data.sleep.training_items || 0} 条可学习的偏好示例已准备好。` : "先在对话中点赞或修正一条回复，再开始聊天学习。") : `${data.narration_train_examples || 0} 条随附叙事示例可用。`;
  $("lastUpdated").textContent=`已同步 ${new Date().toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit"})}`;
  $("connectionBanner").hidden=true;
  renderLearningStatus();
  if (data.job) applySnapshot(data.job);
  renderControls();
  if (state.view==="models") renderModels();
  if (!state.messages.length && (previousReady!==!!server.ready || previousStarting!==server.starting)) renderMessages(false);
}

function renderLearningStatus() {
  if (!state.settings) return;
  const {enabled,auto_sleep,mode,min_signals,idle_seconds}=state.settings;
  const signalCount=state.overview?.store?.new_signals_since_sleep || 0;
  const modes={idle:`空闲 ${Math.round(idle_seconds/60*10)/10} 分钟后学习`,background:"后台短时学习",collect_only:"只收集反馈 · 手动开始学习"};
  $("autoLearnStatus").textContent=!enabled ? "学习信号收集已关闭。启用自动学习后会恢复收集。" : !auto_sleep || mode==="collect_only" ? `只收集反馈 · 手动开始学习。已积累 ${signalCount} / ${min_signals} 条反馈。` : `${modes[mode] || mode}。已积累 ${signalCount} / ${min_signals} 条反馈。`;
}

async function refreshOverview() {
  if (state.pollBusy || !state.authenticated) return;
  state.pollBusy=true;
  try { renderOverview(await api("/api/overview")); }
  catch (error) {
    if (state.authenticated) {
      $("connectionBanner").textContent=`工作空间连接中断：${error.message}。正在自动重试；已保存的反馈和训练记录会保留。`;
      $("connectionBanner").hidden=false;
      $("statusPill").innerHTML="<span></span> 连接中断";
      $("statusPill").className="status-pill disconnected";
      state.overview=null;
      renderControls();
    }
  } finally { state.pollBusy=false; }
}

function syncCharacterForm() {
  $("characterName").value=state.character.name;
  $("systemPrompt").value=state.character.prompt;
  $("temperature").value=state.character.temperature;
  $("maxTokens").value=state.character.max_tokens;
  $("characterLabel").textContent=state.character.name || defaults.name;
}

function renderMessages(scroll=true) {
  if (!state.messages.length) {
    $("messages").innerHTML=`<div class="chat-welcome"><div class="welcome-glyph">✦</div><span class="eyebrow">EVERY STORY STARTS SOMEWHERE</span><h2>今天，想走进哪个故事？</h2><p>${serverReady() ? "描述一个场景，让故事开始。" : state.overview?.server?.starting ? "模型正在加载，首次启动需要一些时间。" : "先启动对话服务，再描述一个场景。"}<br>好的回复点个赞，想改的地方直接编辑。</p>${serverReady() ? "" : `<button class="primary welcome-start" type="button" data-start-server ${state.overview?.server?.starting ? "disabled" : ""}>${state.overview?.server?.starting ? "模型加载中…" : "启动对话服务"} ${icon("arrow")}</button>`}<div class="prompt-suggestions"><button data-prompt="雨夜，我推开一间旅店的门。你是站在柜台后的旅店老板。" type="button"><span>雨夜旅店</span>一段安静的相遇 ${icon("arrow")}</button><button data-prompt="你是星际飞船的领航员。我们刚收到一个陌生星球发来的求救信号。" type="button"><span>星际旅程</span>让故事从未知开始 ${icon("arrow")}</button><button data-prompt="你是我有点嘴硬但很可靠的同学。课间，我们一起聊聊今天的课程。" type="button"><span>课间日常</span>和角色随意聊两句 ${icon("arrow")}</button></div></div>`;
  } else {
    $("messages").innerHTML=state.messages.map((m,index) => `<article class="message message-${m.role}"><div class="message-avatar">${m.role==="assistant" ? "✦" : "你"}</div><div class="message-body"><div class="message-meta"><strong>${m.role==="assistant" ? esc(state.character.name) : "你"}</strong><span>${m.role==="assistant" ? esc(m.adapter || "Roleplay") : ""}</span></div><div class="message-text">${esc(m.content)}</div>${m.role==="assistant" && m.gen_id ? `<div class="message-actions"><button class="text-button${m.feedback==="good" ? " selected" : ""}" data-feedback="good" data-index="${index}" ${state.chatBusy ? "disabled" : ""} aria-label="喜欢这条回复">${icon("up")} <span>${m.feedback==="good" ? "已喜欢" : "喜欢"}</span></button><button class="text-button${m.feedback==="bad" ? " selected" : ""}" data-feedback="bad" data-index="${index}" ${state.chatBusy ? "disabled" : ""} aria-label="不喜欢这条回复">${icon("down")} <span>${m.feedback==="bad" ? "已标记" : "不喜欢"}</span></button><button class="text-button" data-edit="${index}" ${state.chatBusy ? "disabled" : ""}>${icon("edit")} ${m.feedback==="fix" ? "已修正 · 编辑" : "修改并教给模型"}</button>${index===state.messages.length-1 ? `<button class="text-button" data-regenerate="${index}" ${state.chatBusy ? "disabled" : ""}>${icon("refresh")} 重新生成</button>` : ""}</div><div class="correction-editor" id="edit-${index}" hidden><label for="correction-${index}">写下你更喜欢的回复</label><textarea id="correction-${index}" rows="5" maxlength="16000">${esc(m.content)}</textarea><div class="btn-row"><button class="primary" data-save-edit="${index}">保存修正</button><button class="secondary" data-cancel-edit="${index}">取消</button></div></div>` : ""}</div></article>`).join("");
  }
  if (state.chatBusy) $("messages").insertAdjacentHTML("beforeend",`<div class="message pending-message"><div class="message-avatar">✦</div><div class="message-body"><div class="thinking"><i></i><i></i><i></i></div><span>正在生成回复，CPU 上需要一些时间…</span></div></div>`);
  if (scroll) $("messages").scrollTop=state.messages.length ? $("messages").scrollHeight : 0;
}

async function sendMessage(regenerate=false) {
  if (state.chatBusy || !serverReady()) { if (!serverReady()) toast("请先启动对话服务，等待模型就绪。",true); return; }
  const input=$("chatInput").value.trim();
  if (!regenerate && !input) return;
  if (regenerate && state.messages.at(-1)?.role!=="assistant") return;
  const original=[...state.messages];
  if (regenerate) state.messages.pop();
  else { state.messages.push({role:"user",content:input}); $("chatInput").value=""; }
  state.chatBusy=true;
  $("chatError").hidden=true;
  renderMessages();
  renderControls();
  try {
    const messages=[{role:"system",content:state.character.prompt},...state.messages.map(({role,content}) => ({role,content}))];
    const data=await post("/api/chat",{messages,session_id:state.session,temperature:Number(state.character.temperature),max_tokens:Number(state.character.max_tokens)});
    const content=data.choices?.[0]?.message?.content ?? data.content;
    if (typeof content!=="string" || !content.trim()) throw new Error("模型返回了空回复。请重新生成，或降低最长回复长度后重试。");
    state.messages.push({role:"assistant",content,gen_id:data.rp_generation_id || data.generation_id,adapter:data.rp_adapter || data.adapter || state.overview?.adapter,created_at:Date.now()});
    saveChat();
  } catch (error) {
    state.messages=original;
    if (!regenerate) $("chatInput").value=input;
    $("chatError").textContent=`回复未完成：${error.message} 你的输入已保留。`;
    $("chatError").hidden=false;
  } finally { state.chatBusy=false; renderMessages(); renderControls(); refreshOverview(); }
}

async function feedback(index,rating,correction) {
  const message=state.messages[index];
  if (!message?.gen_id) throw new Error("这条回复没有可追踪的生成记录。");
  if (correction!==undefined && !correction.trim()) throw new Error("修正内容不能为空。");
  await post("/api/feedback",{gen_id:message.gen_id,rating,...(correction!==undefined ? {correction:correction.trim()} : {})});
  message.feedback=correction!==undefined ? "fix" : rating;
  if (correction!==undefined) message.content=correction.trim();
  saveChat();
  renderMessages(false);
  toast(correction!==undefined ? "修正已保存。下次学习会比较新回复与原回复。" : "反馈已保存。你可以在学习记录中审核。");
  refreshOverview();
}

function renderChart(metrics) {
  const points=metrics.map((m,i) => ({step:Number(m.step ?? i+1),loss:Number(m.loss ?? m.train_loss)})).filter((m) => Number.isFinite(m.loss));
  if (!points.length) { $("lossChart").innerHTML="<div class='chart-empty'>尚无损失记录。准备或评估阶段不会生成训练曲线。</div>"; return; }
  const width=580,height=170,left=46,right=14,top=15,bottom=30;
  let min=Math.min(...points.map((p)=>p.loss)),max=Math.max(...points.map((p)=>p.loss));
  const spread=Math.max(max-min,.01); min=Math.max(0,min-spread*.1);max+=spread*.1;
  const stepMin=points[0].step,stepMax=Math.max(stepMin+1,points.at(-1).step);
  const x=(step)=>left+(step-stepMin)/(stepMax-stepMin)*(width-left-right);
  const y=(loss)=>top+(max-loss)/(max-min)*(height-top-bottom);
  const path=points.map((p,i)=>`${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join(" ");
  const grid=[0,.5,1].map((v)=>{const val=min+(max-min)*v,yy=y(val);return `<line x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}" class="chart-grid"/><text x="${left-8}" y="${yy+4}" text-anchor="end">${val.toFixed(2)}</text>`;}).join("");
  $("lossChart").innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="训练损失曲线，${points.length} 个实际记录点，最新损失 ${points.at(-1).loss.toFixed(4)}"><defs><linearGradient id="lossShade" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#83bba6" stop-opacity=".2"/><stop offset="100%" stop-color="#83bba6" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${path} L${x(points.at(-1).step)},${height-bottom} L${x(points[0].step)},${height-bottom} Z" fill="url(#lossShade)"/><path d="${path}" fill="none" stroke="#94c4b0" stroke-width="2.5" stroke-linejoin="round"/>${points.length===1 ? `<circle cx="${x(points[0].step)}" cy="${y(points[0].loss)}" r="3" fill="#94c4b0"/>` : ""}<text x="${left}" y="${height-6}">Step ${stepMin}</text><text x="${width-right}" y="${height-6}" text-anchor="end">Step ${points.at(-1).step}</text></svg>`;
}

function applySnapshot(snapshot) {
  state.job=snapshot;
  if (state.inspectedJob && state.inspectedJob.id===snapshot.id) state.inspectedJob=null;
  renderJob(state.inspectedJob || snapshot);
  renderControls();
}

function renderJob(job) {
  if (!job) return;
  const progress=Math.max(0,Math.min(1,Number(job.progress)||0));
  $("jobBadge").textContent=jobState(job.state);
  $("jobBadge").className=`badge${["done","completed"].includes(job.state) ? " badge-success" : ["error","failed"].includes(job.state) ? " badge-error" : ""}`;
  $("jobTitle").textContent=`${jobKind(job.kind)}${job.id ? ` · ${job.id.slice(0,12)}` : ""}`;
  const baseModel=job.result?.base_model || job.config?.model;
  if ($("jobModel")) {
    $("jobModel").hidden=!baseModel;
    $("jobModel").textContent=baseModel ? `基础模型 · ${baseModel}` : "";
  }
  $("progressFill").style.width=`${progress*100}%`;
  $("progressText").textContent=`${Math.round(progress*100)}%`;
  const metrics=Array.isArray(job.metrics) ? job.metrics : [];
  const latest=metrics.at(-1) || {};
  const lastValue=(...names)=>{for(let i=metrics.length-1;i>=0;i--) for(const name of names) if(metrics[i][name]!=null) return metrics[i][name];return null;};
  const step=lastValue("step"),totalSteps=lastValue("total_steps"),loss=lastValue("loss"),learningRate=lastValue("learning_rate"),throughput=lastValue("tokens_per_s","tokens_per_second");
  $("metricSteps").textContent=step!=null ? `${step}${totalSteps ? ` / ${totalSteps}` : ""}` : "—";
  $("metricLoss").textContent=loss!=null ? fmt(loss,4) : "—";
  $("metricLearningRate").textContent=learningRate!=null ? Number(learningRate).toExponential(2) : "—";
  $("metricEta").textContent=fmtTime(lastValue("eta_s"));
  $("metricThroughput").textContent=throughput!=null ? fmt(throughput,1) : "—";
  let start=typeof job.started_at==="number" ? job.started_at*1000 : Date.parse(job.started_at);
  let finish=typeof job.finished_at==="number" ? job.finished_at*1000 : Date.parse(job.finished_at);
  $("metricElapsed").textContent=fmtTime(latest.elapsed_s ?? (Number.isFinite(start) ? ((Number.isFinite(finish) ? finish : Date.now())-start)/1000 : NaN));
  renderChart(metrics);
  $("log").textContent=job.log_tail?.join("\n") || "尚无训练日志。";
  $("logLineCount").textContent=`${job.log_tail?.length || 0} 行`;
  const checkpoint=job.checkpoint;
  $("checkpointLine").textContent=checkpoint ? `检查点：${typeof checkpoint==="string" ? checkpoint : checkpoint.path || checkpoint.version || "已保存"}` : "尚无检查点";
  $("jobError").hidden=!job.error;
  $("jobError").textContent=job.error || "";
  $("summaryBody").hidden=!job.result || job.state==="paused";
  if (job.result) {
    const result=job.result;
    if (job.kind==="sleep") $("summaryBody").innerHTML=`<strong>${result.status==="skipped" ? "没有可学习的新数据" : result.activated ? "新聊天版本已通过检查" : "训练已结束，版本尚未激活"}</strong><p>${esc(result.reason || result.version || "")}</p><div class="result-pairs"><span>偏好 NLL</span><strong>${fmt(result.train_nll_chosen_before,4)} → ${fmt(result.train_nll_chosen_after,4)}</strong><span>能力漂移</span><strong>${fmt(result.canary_drift,4)} ${result.guard_passed===true ? "· 检查通过" : result.guard_passed===false ? "· 检查未通过" : ""}</strong></div>`;
    else if (result.eval_md) $("summaryBody").innerHTML=`<strong>模型对比结果</strong><pre class="evaluation-report">${esc(result.eval_md)}</pre>`;
    else $("summaryBody").innerHTML=`<strong>${result.status==="ok" ? "叙事适配器已保存" : "训练结果"}</strong><div class="result-pairs"><span>训练损失</span><strong>${fmt(result.train_nll_mean,4)}</strong><span>示例数</span><strong>${esc(result.examples ?? "—")}</strong><span>训练分钟</span><strong>${fmt(result.train_minutes,2)}</strong></div>${result.out ? `<code>${esc(result.out)}</code>` : ""}`;
  }
}

async function loadHistory() {
  try {
    const data=await api("/api/jobs");
    state.history=Array.isArray(data) ? data : data.items || data.jobs || [];
    $("jobHistory").className=state.history.length ? "history-list" : "empty-state compact";
    $("jobHistory").innerHTML=state.history.length ? state.history.map((job) => `<button class="history-row" data-job="${esc(job.id)}"><span class="history-icon">${icon("train")}</span><span><strong>${esc(jobKind(job.kind))}</strong><small>${esc(job.id)} · ${job.created_at ? esc(new Date(typeof job.created_at==="number" ? job.created_at*1000 : job.created_at).toLocaleString("zh-CN")) : ""}</small></span><span class="badge">${esc(jobState(job.state))}</span><span class="small muted">${Math.round((job.progress || 0)*100)}%</span>${icon("arrow")}</button>`).join("") : "完成第一轮训练后，会在这里保留记录。";
  } catch(error) { if (state.view==="training") toast(error.message,true); }
}

function trainingPayload() {
  const payload={preset:state.preset};
  if (document.querySelector(".advanced-settings").open) Object.assign(payload,{threads:Number($("trainThreads").value),max_minutes:Number($("trainMinutes").value),epochs:Number($("trainEpochs").value)});
  if ($("trainingType").value==="narration") { payload.data=$("trainingData").value; if (document.querySelector(".advanced-settings").open) payload.lr=Number($("trainLr").value); }
  return payload;
}

function renderPresetSummary() {
  const advanced=document.querySelector(".advanced-settings");
  advanced.querySelectorAll("input").forEach((input)=>input.disabled=!advanced.open);
  $("trainLr").step="any";
  const payload=trainingPayload();
  const preset={quick:[1,5,1],balanced:[2,15,2],thorough:[4,45,4]}[state.preset];
  $("presetSummary").textContent=`${payload.threads ?? preset[0]} CPU 线程 · 最多 ${payload.max_minutes ?? preset[1]} 分钟 · ${payload.epochs ?? preset[2]} 轮`;
}

async function startTraining(type, useSettings=false) {
  const payload=useSettings ? {preset:state.preset,threads:state.settings?.threads || 1,max_minutes:state.settings?.max_minutes || 5,epochs:state.settings?.epochs || 2} : trainingPayload();
  const snapshot=await post(type==="narration" ? "/api/narration/train" : "/api/sleep",payload);
  state.inspectedJob=null;
  applySnapshot(snapshot);
  navigate("training");
  toast("任务已创建。准备阶段可能需要下载模型，请查看运行日志。");
  await refreshOverview();
}

function contextForSignal(signal) {
  try {
    const messages=typeof signal.messages==="string" ? JSON.parse(signal.messages) : signal.messages;
    if (!Array.isArray(messages)) return "未记录上下文";
    return messages.map((message)=>`${({system:"角色设定",user:"用户",assistant:"模型"})[message.role] || message.role}: ${message.content}`).join("\n\n");
  } catch { return "上下文格式无法读取"; }
}

async function loadSignals() {
  try {
    const data=await api(`/api/signals?page=${state.signalPage}&page_size=12`);
    state.signals=data.items || [];
    state.signalTotal=data.total || 0;
    $("signalTotal").textContent=`${state.signalTotal} 条学习记录`;
    $("signalsList").innerHTML=state.signals.length ? state.signals.map((signal) => `<details class="signal-item${signal.active ? "" : " signal-excluded"}"><summary><span class="signal-icon">${icon(signal.kind==="bad" ? "down" : signal.kind==="fix" || signal.kind==="edit" ? "edit" : "up")}</span><span class="signal-preview"><strong>${esc(signalKind(signal.kind))} <span class="small muted">#${esc(signal.id)}</span></strong><span>${esc((signal.chosen || signal.rejected || "空记录").slice(0,110))}</span></span><span class="badge${signal.active ? " badge-success" : ""}">${signal.active ? signal.consumed_in ? `已用于 ${esc(signal.consumed_in)}` : "待学习" : "已排除"}</span></summary><div class="signal-content"><details class="signal-context"><summary>查看角色设定与对话上下文</summary><pre>${esc(contextForSignal(signal))}</pre></details><div class="signal-comparison"><div><span class="eyebrow">CHOSEN · 更喜欢的回复</span><p>${esc(signal.chosen || "此反馈没有正向示例")}</p></div><div><span class="eyebrow">REJECTED · 不喜欢的回复</span><p>${esc(signal.rejected || "此反馈没有负向示例")}</p></div></div><div class="signal-footer"><span class="small muted">${signal.consumed_in ? "已使用的信号不能撤销已完成训练中的影响。" : "排除后不会进入后续训练，记录仍会保留。"}</span><button class="secondary" data-signal-toggle="${esc(signal.id)}" data-active="${signal.active ? "false" : "true"}">${signal.active ? "从后续学习中排除" : "重新纳入学习"}</button></div></div></details>`).join("") : `<div class="empty-state"><span class="empty-glyph">${icon("learn")}</span><h2>从一条有用的反馈开始。</h2><p>在对话里点赞、标记不喜欢或修正回复。<br>这里会保留原文与偏好，供你审核。</p><button class="primary" data-go="chat">开始对话 ${icon("arrow")}</button></div>`;
    $("signalsPrevBtn").disabled=state.signalPage<=1;
    $("signalsNextBtn").disabled=state.signalPage*12>=state.signalTotal;
    $("signalsPage").textContent=`第 ${state.signalPage} / ${Math.max(1,Math.ceil(state.signalTotal/12))} 页`;
  } catch(error) { if (state.view==="learning") toast(error.message,true); }
}

function renderModels() {
  const data=state.overview;
  if (!data) { $("modelsList").innerHTML="<div class='empty-state'>等待工作空间连接…</div>"; return; }
  const models=[{version:"base",base:true},...(data.adapters || [])];
  $("activeModelBadge").textContent=`当前选择 · ${data.adapter || "base"}`;
  $("modelsList").innerHTML=models.map((model) => {
    const version=model.version,active=version===data.adapter;
    const title=model.base ? "基础模型" : version==="narration" ? "叙事适配器" : `聊天版本 ${version}`;
    const description=model.base ? "未经角色扮演微调的原始模型，用于对照和回退。" : version==="narration" ? "学习场景描写、动作与对话的平衡，作为聊天学习基础。" : "从聊天偏好学习的新版本，保留对应的叙事基础。";
    return `<article class="card model-card${active ? " model-active" : ""}"><div class="card-heading"><span class="model-symbol">${icon("model")}</span><span class="badge${active ? " badge-success" : ""}">${active ? "当前选择" : esc(version)}</span></div><h2>${esc(title)}</h2><p class="hint">${description}</p><div class="model-meta"><span>版本</span><code>${esc(version)}</code>${model.train_minutes!=null ? `<span>训练耗时</span><strong>${fmt(model.train_minutes,2)} 分钟</strong>` : ""}${model.n_train!=null || model.examples!=null ? `<span>学习示例</span><strong>${esc(model.n_train ?? model.examples)}</strong>` : ""}${model.guard_passed!=null ? `<span>质量检查</span><strong>${model.guard_passed ? "通过" : "未通过"}</strong>` : ""}</div>${model.legacy_foundation ? "<p class='small warning-text'>旧版本未记录固定叙事基础，加载时使用兼容模式。</p>" : ""}<div class="model-actions"><button class="${active ? "secondary" : "primary"}" data-activate="${esc(version)}" ${active || isRunning() || state.chatBusy ? "disabled" : ""}>${active ? "已选择" : "选择此版本"}</button>${model.base ? "" : `<a class="secondary" href="/api/adapters/${encodeURIComponent(version)}/export" download="cradle-${esc(version)}.zip" aria-label="导出 ${esc(version)}">${icon("download")} 导出</a>`}</div></article>`;
  }).join("");
}

async function download(path,filename) {
  const response=await fetch(path,{credentials:"same-origin"});
  if (response.status===401) { showAuth(); throw new Error("请先连接工作空间。"); }
  if (!response.ok) { const data=await response.json().catch(()=>({})); throw new Error(data.error || "导出失败，请重试。"); }
  const blob=await response.blob();
  saveDownload(blob,filename);
}

function saveDownload(blob,filename) {
  const url=URL.createObjectURL(blob),link=document.createElement("a");
  link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

async function loadSettings() {
  const settings=await api("/api/settings");
  state.settings=settings;
  $("autoLearnToggle").checked=!!settings.enabled && !!settings.auto_sleep && settings.mode!=="collect_only";
  $("settingsEnabled").checked=$("autoLearnToggle").checked;
  $("learningMode").value=settings.mode;
  $("minSignals").value=settings.min_signals;
  $("idleSeconds").value=settings.idle_seconds;
  $("maxMinutes").value=settings.max_minutes;
  $("settingsThreads").value=settings.threads;
  $("settingsEpochs").value=settings.epochs;
  $("guardDrift").value=settings.guard_max_drift;
  renderModeHint();
  renderLearningStatus();
}

function renderModeHint() {
  $("modeHint").textContent=({collect_only:"只收集反馈；手动点击「现在学习」时才训练，适合上课和低电量时。",idle:"积累足够反馈并停止对话一段时间后，自动进行短时学习。",background:"积累足够反馈后触发短时训练，会占用 CPU；对话与训练会按服务能力协调。"})[$("learningMode").value];
}

function connectEvents() {
  if (!state.authenticated) return;
  state.events?.close();
  state.events=new EventSource("/api/events");
  state.events.onmessage=(event)=>{
    try {
      const message=JSON.parse(event.data);
      if (message.type==="snapshot") applySnapshot(message.data);
      if (message.type==="state" && message.snapshot) { applySnapshot(message.snapshot); refreshOverview(); if (state.view==="training") loadHistory(); }
      if (message.type==="log" && state.job) {
        const logs=[...(state.job.log_tail || []),message.line].slice(-300);
        const snapshot={...state.job,log_tail:logs};
        if (message.progress!=null) snapshot.progress=message.progress;
        if (message.result) snapshot.result=message.result;
        applySnapshot(snapshot);
      }
    } catch { }
  };
}

function newChat() {
  state.messages=[];state.session=uid();$("chatError").hidden=true;saveChat();renderMessages();$("chatInput").focus();
}

function bindEvents() {
  document.addEventListener("click",(event)=>{
    const nav=event.target.closest("[data-view],[data-go]");
    if (nav) navigate(nav.dataset.view || nav.dataset.go);
    if (event.target.closest("[data-start-server]")) $("serverStartBtn").click();
    const prompt=event.target.closest("[data-prompt]");
    if (prompt) { $("chatInput").value=prompt.dataset.prompt;$("chatInput").focus(); }
  });
  $("newChatBtn").addEventListener("click",async()=>{ if (!state.messages.length || await confirmAction("开始新对话？当前浏览器中的对话将被替换；已有学习反馈会保留。")) newChat(); });
  $("chatForm").addEventListener("submit",(event)=>{event.preventDefault();sendMessage();});
  $("chatInput").addEventListener("keydown",(event)=>{if (event.key==="Enter" && (event.ctrlKey || event.metaKey)) {event.preventDefault();sendMessage();}});
  $("openCharacterBtn").addEventListener("click",()=>{navigate("settings");$("characterName").focus();});
  $("messages").addEventListener("click",(event)=>{
    const button=event.target.closest("button");if (!button) return;
    if (button.dataset.feedback) action(button,()=>feedback(Number(button.dataset.index),button.dataset.feedback));
    if (button.dataset.regenerate) sendMessage(true);
    if (button.dataset.edit) {const index=button.dataset.edit;$("edit-"+index).hidden=false;$("correction-"+index).focus();}
    if (button.dataset.cancelEdit) $("edit-"+button.dataset.cancelEdit).hidden=true;
    if (button.dataset.saveEdit) {const index=Number(button.dataset.saveEdit);action(button,()=>feedback(index,"good",$("correction-"+index).value));}
  });
  $("serverStartBtn").addEventListener("click",()=>action($("serverStartBtn"),async()=>{await post("/api/server/start");toast("对话服务正在启动。请等待模型加载完成。");await refreshOverview();}));
  $("serverStopBtn").addEventListener("click",async()=>{if (await confirmAction("停止对话服务？正在使用的 SillyTavern 连接也会断开。")) action($("serverStopBtn"),async()=>{await post("/api/server/stop");await refreshOverview();});});
  $("serverLogBtn").addEventListener("click",()=>action($("serverLogBtn"),()=>download("/api/server/logs","cradle-server.log")));
  $("copyEndpointBtn").addEventListener("click",()=>action($("copyEndpointBtn"),async()=>{await navigator.clipboard.writeText($("serverEndpoint").textContent);toast("连接地址已复制。");}));
  $("autoLearnToggle").addEventListener("change",()=>action(null,async()=>{try {const active=$("autoLearnToggle").checked;state.settings=await patch("/api/settings",{enabled:true,auto_sleep:active,...(active && state.settings?.mode==="collect_only" ? {mode:"idle"} : {})});$("settingsEnabled").checked=!!state.settings.auto_sleep;$("learningMode").value=state.settings.mode;renderModeHint();renderLearningStatus();toast(state.settings.auto_sleep ? "自动学习已启用。" : "自动学习已关闭，反馈仍会保留。");} catch(error) {$("autoLearnToggle").checked=!!state.settings?.enabled && !!state.settings?.auto_sleep && state.settings?.mode!=="collect_only";throw error;}}));
  $("learnNowBtn").addEventListener("click",()=>action($("learnNowBtn"),()=>startTraining("sleep",true)));
  $("trainingType").addEventListener("change",()=>{$("narrationDataControl").hidden=$("trainingType").value!=="narration";$("trainingTypeHint").textContent=$("trainingType").value==="narration" ? "从随附叙事数据学习人物对白与场景描写。" : "从已审核的点赞和修改中学习，保持原有叙事基础。";if (state.overview) renderOverview(state.overview);});
  document.querySelectorAll("[data-preset]").forEach((button)=>button.addEventListener("click",()=>{state.preset=button.dataset.preset;document.querySelectorAll("[data-preset]").forEach((b)=>b.classList.toggle("active",b===button));const values={quick:[1,5,1],balanced:[2,15,2],thorough:[4,45,4]}[state.preset];$("trainThreads").value=values[0];$("trainMinutes").value=values[1];$("trainEpochs").value=values[2];renderPresetSummary();}));
  document.querySelector(".advanced-settings").addEventListener("toggle",renderPresetSummary);
  ["trainThreads","trainMinutes","trainEpochs"].forEach((id)=>$(id).addEventListener("input",renderPresetSummary));
  $("trainingForm").addEventListener("submit",(event)=>{event.preventDefault();action($("startTrainingBtn"),()=>startTraining($("trainingType").value));});
  $("evalNarrBtn").addEventListener("click",()=>action($("evalNarrBtn"),async()=>{state.inspectedJob=null;applySnapshot(await post("/api/narration/eval"));navigate("training");}));
  $("stopJobBtn").addEventListener("click",async()=>{if (await confirmAction("保存检查点并停止当前任务？保存结束前请保持应用运行。")) action($("stopJobBtn"),async()=>{await post("/api/job/stop");toast("已请求停止，正在尝试保存检查点。");await refreshOverview();});});
  $("resumeJobBtn").addEventListener("click",()=>action($("resumeJobBtn"),async()=>{const job=state.inspectedJob || state.job;const snapshot=await post("/api/job/resume",{job_id:job?.id});state.inspectedJob=null;applySnapshot(snapshot);await refreshOverview();toast("已创建恢复任务，将从保存的检查点继续。");}));
  $("downloadLogBtn").addEventListener("click",()=>action($("downloadLogBtn"),()=>{const job=state.inspectedJob || state.job;return download(`/api/jobs/${encodeURIComponent(job.id)}/logs`,`cradle-${job.id}.log`);}));
  $("refreshJobsBtn").addEventListener("click",()=>action($("refreshJobsBtn"),loadHistory));
  $("jobHistory").addEventListener("click",(event)=>{const button=event.target.closest("[data-job]");if (!button) return;state.inspectedJob=state.history.find((j)=>j.id===button.dataset.job);renderJob(state.inspectedJob);renderControls();$("view-training").scrollIntoView({behavior:"smooth",block:"start"});});
  $("refreshSignalsBtn").addEventListener("click",()=>action($("refreshSignalsBtn"),loadSignals));
  $("signalsPrevBtn").addEventListener("click",()=>{state.signalPage--;loadSignals();});
  $("signalsNextBtn").addEventListener("click",()=>{state.signalPage++;loadSignals();});
  $("signalsList").addEventListener("click",(event)=>{const button=event.target.closest("[data-signal-toggle]");if (button) action(button,async()=>{await patch(`/api/signals/${encodeURIComponent(button.dataset.signalToggle)}`,{active:button.dataset.active==="true"});await loadSignals();await refreshOverview();});});

  $("modelsList").addEventListener("click",(event)=>{const button=event.target.closest("button");if (!button) return;if (button.dataset.activate) action(button,async()=>{await post("/api/adapters/activate",{version:button.dataset.activate});await refreshOverview();toast("模型版本已选择。运行中的服务会按后端状态重新加载。");});});
  $("learningMode").addEventListener("change",()=>{renderModeHint();$("settingsEnabled").checked=$("learningMode").value!=="collect_only";});
  $("settingsForm").addEventListener("submit",(event)=>{event.preventDefault();action($("saveSettingsBtn"),async()=>{state.settings=await patch("/api/settings",{enabled:true,auto_sleep:$("settingsEnabled").checked && $("learningMode").value!=="collect_only",mode:$("learningMode").value,min_signals:Number($("minSignals").value),idle_seconds:Number($("idleSeconds").value),max_minutes:Number($("maxMinutes").value),threads:Number($("settingsThreads").value),epochs:Number($("settingsEpochs").value),guard_max_drift:Number($("guardDrift").value)});$("autoLearnToggle").checked=!!state.settings.enabled && !!state.settings.auto_sleep && state.settings.mode!=="collect_only";$("settingsEnabled").checked=$("autoLearnToggle").checked;renderLearningStatus();toast("学习设置已保存，会在下次训练时生效。");});});
  $("saveCharacterBtn").addEventListener("click",async()=>{
    if (state.chatBusy) return;
    const temperature=Number($("temperature").value),maxTokens=Number($("maxTokens").value);
    if (temperature<0 || temperature>2 || !Number.isFinite(temperature) || !Number.isInteger(maxTokens) || maxTokens<16 || maxTokens>1024) {toast("创造性需为 0–2，最长回复需为 16–1024 的整数。",true);return;}
    const character={name:$("characterName").value.trim() || defaults.name,prompt:$("systemPrompt").value.trim() || defaults.prompt,temperature,max_tokens:maxTokens};
    const changed=character.name!==state.character.name || character.prompt!==state.character.prompt;
    if (changed && state.messages.length && !await confirmAction("更改角色会开始新对话，以保持会话学习信号独立。已有反馈会保留。")) return;
    state.character=character;if (changed) newChat();saveChat();syncCharacterForm();toast("角色设定已保存。");
  });
  $("clearChatBtn").addEventListener("click",async()=>{if (state.chatBusy) {toast("请先等待当前回复生成结束。",true);return;}if (await confirmAction("清除当前浏览器中的对话？后端学习反馈和训练记录会保留。")) {newChat();toast("浏览器中的当前对话已清除。");}});
  $("authDialog").addEventListener("cancel",(event)=>event.preventDefault());
  $("authForm").addEventListener("submit",async(event)=>{event.preventDefault();$("loginBtn").disabled=true;$("authError").textContent="";try {await post("/api/auth/login",{token:$("accessToken").value});state.authenticated=true;$("accessToken").value="";$("authDialog").close();await loadSettings();connectEvents();await refreshOverview();}catch(error){$("authError").textContent=error.message;}finally{$("loginBtn").disabled=false;}});
  $("logoutBtn").addEventListener("click",()=>action($("logoutBtn"),async()=>{await post("/api/auth/logout");showAuth();}));
  window.addEventListener("hashchange",()=>navigate(location.hash.slice(1)));
  window.addEventListener("beforeunload",()=>state.events?.close());
}

async function initialize() {
  restoreChat();syncCharacterForm();renderMessages();bindEvents();renderPresetSummary();navigate(location.hash.slice(1) || "chat");
  await refreshOverview();
  if (state.authenticated) {
    try {await loadSettings();connectEvents();} catch(error) {if (state.authenticated) toast(error.message,true);}
  }
  setInterval(()=>{if (!document.hidden) refreshOverview();},4000);
  document.addEventListener("visibilitychange",()=>{if (!document.hidden) refreshOverview();});
}

initialize();
