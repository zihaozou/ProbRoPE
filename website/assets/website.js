(() => {
  'use strict';
  const dialog = document.querySelector('#citation-dialog');
  document.querySelector('#citation-open').addEventListener('click', () => dialog.showModal());
  document.querySelector('#citation-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    const box = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
  });
  document.querySelector('#citation-copy').addEventListener('click', async () => {
    const status = document.querySelector('#copy-status');
    try {
      await navigator.clipboard.writeText(document.querySelector('#citation-text').textContent);
      status.textContent = 'BibTeX copied to clipboard.';
    } catch {
      status.textContent = 'Select the citation above to copy it manually.';
    }
  });

  // Keep all content readable if GSAP fails to load or JavaScript is disabled.
  if (!window.gsap || !window.ScrollToPlugin) return;
  gsap.registerPlugin(ScrollToPlugin);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const touchViewport=matchMedia('(hover:none) and (pointer:coarse)');
  function lockTouchViewport(){
    if(!touchViewport.matches)return;
    const root=document.documentElement;
    root.style.removeProperty('--page-height');
    root.style.removeProperty('--browser-inset');
    const measured=document.querySelector('.paper-page').clientHeight;
    const height=window.innerWidth<=760?Math.max(measured,window.screen.height):measured;
    root.style.setProperty('--content-height',`${window.innerHeight}px`);
    root.style.setProperty('--page-height',`${height}px`);
    root.style.setProperty('--browser-inset',`${Math.max(0,height-window.innerHeight)}px`);
  }
  lockTouchViewport();
  let stops = [0];
  let current = 0;
  let tween;
  let scrollFrame;
  let settleTimer;
  let scripted = false;
  const pageNav=document.querySelector('.page-nav');
  const fusionContent=[...document.querySelectorAll('.fusion-image,.fusion-copy h2,.fusion-copy p,.fusion-source')];
  let fusionEntered=false;
  let fusionEntrance;
  gsap.set(fusionContent,{opacity:0,y:index=>index===0?0:28});
  function revealFusionContent(lift){
    if(lift<=1){
      if(fusionEntered){
        fusionEntrance?.kill();
        fusionEntered=false;
        gsap.set(fusionContent,{opacity:0,y:index=>index===0?0:28});
      }
    }else if(lift>window.innerHeight*.5 && !fusionEntered){
      fusionEntered=true;
      fusionEntrance=gsap.timeline()
        .to(fusionContent[0],{opacity:1,duration:reducedMotion.matches?0:.9,ease:'power1.inOut',overwrite:true},0)
        .to(fusionContent.slice(1),{opacity:1,y:0,duration:reducedMotion.matches?0:.75,
          stagger:reducedMotion.matches?0:.09,ease:'power3.out',overwrite:true},reducedMotion.matches?0:.08);
    }
  }
  const curtainFlight={active:false,velocity:0,pixelsPerSecond:0};
  function animateCurtain(destination,velocity){
    const boundary=stops[stops.length-2];
    const base=destination?Math.min(window.scrollY,boundary):boundary;
    const height=stops[stops.length-1]-base;
    const longCurtain=height>window.innerHeight*1.5;
    const clockScale=longCurtain?.6:1;
    const rebound=.14*Math.min(1,window.innerHeight/height);
    let position=Math.max(0,Math.min(1,(window.scrollY-base)/height));
    let elapsed=0;
    curtainFlight.active=true;
    curtainFlight.velocity=velocity/(height*clockScale);
    function tick(time,deltaMs){
      const duration=Math.min(deltaMs/1000,.04)*clockScale;
      const steps=Math.max(1,Math.ceil(duration*240)),dt=duration/steps;
      let finished=false;
      for(let i=0;i<steps;i++){
        const acceleration=destination ? 28*(1-position)-3*curtainFlight.velocity : -6.5;
        curtainFlight.velocity+=acceleration*dt;
        position+=curtainFlight.velocity*dt;
        elapsed+=dt;
        if(position>=1){
          position=1;
          if(destination && Math.abs(curtainFlight.velocity)<.65){finished=true;break;}
          curtainFlight.velocity=-Math.abs(curtainFlight.velocity)*rebound;
        }else if(position<=0){
          position=0;
          if(!destination && Math.abs(curtainFlight.velocity)<.65){finished=true;break;}
          curtainFlight.velocity=Math.abs(curtainFlight.velocity)*.14;
        }
        if(elapsed>1.4){finished=true;break;}
      }
      curtainFlight.pixelsPerSecond=curtainFlight.velocity*height*clockScale;
      window.scrollTo(0,base+(finished?destination:position)*height);
      activatePage();
      if(finished){cancelJump();activatePage();}
    }
    tween={kill(){gsap.ticker.remove(tick);curtainFlight.active=false;}};
    gsap.ticker.add(tick);
  }
  const navButtons=[...pageNav.querySelectorAll('[data-goto]')];
  const svgNS='http://www.w3.org/2000/svg';
  const indicator=document.createElementNS(svgNS,'svg');
  indicator.classList.add('page-indicator'); indicator.setAttribute('aria-hidden','true');
  const ball=document.createElementNS(svgNS,'path'); ball.setAttribute('fill','#262626');
  indicator.append(ball); pageNav.prepend(indicator); pageNav.classList.add('animated-nav');
  // Position and deformation have separate inertia. Changing the destination
  // preserves both velocities, including when a user reverses mid-flight.
  const soft={x:0,v:0,lag:0,lagV:0,target:0,y:0,r:22};
  let navIndex=-1, ticking=false;
  function paintIndicator() {
    const {x,y,r,lag}=soft;
    const direction=lag<0?-1:1;
    const pull=Math.min(Math.abs(lag),20), deformation=Math.min(1,pull/12);
    const h=r/Math.sqrt(1+pull/r*.85), k=.55228475;
    // Work in a travel-aligned coordinate system. The widest section moves
    // toward the rounded nose; the rear arcs taper to a narrow, soft tail.
    const shoulder=pull*.45;
    const nose=r+pull*.16, tail=-r-pull*1.2;
    const world=u=>Math.max(1,Math.min(pageNav.clientWidth-1,x+direction*u));
    const shoulderX=world(shoulder), noseX=world(nose), tailX=world(tail);
    const tailTangent=h*k*(1-.94*deformation);
    const rearControl=world(shoulder-(shoulder-tail)*(k+.12*deformation));
    const tipControl=world(tail+r*.06*deformation);
    ball.setAttribute('d',`M ${shoulderX} ${y-h}
      C ${world(shoulder+(nose-shoulder)*k)} ${y-h} ${noseX} ${y-h*k} ${noseX} ${y}
      C ${noseX} ${y+h*k} ${world(shoulder+(nose-shoulder)*k)} ${y+h} ${shoulderX} ${y+h}
      C ${rearControl} ${y+h} ${tipControl} ${y+tailTangent} ${tailX} ${y}
      C ${tipControl} ${y-tailTangent} ${rearControl} ${y-h} ${shoulderX} ${y-h} Z`);
    const left=Math.min(tailX,noseX), right=Math.max(tailX,noseX);
    navButtons.forEach(button=>{
      const center=button.offsetLeft+button.offsetWidth/2;
      button.style.color=center>=left && center<=right ? '#fff' : '#6d6b68';
    });
  }
  function tickBall(time,deltaMs) {
    // Substeps keep the spring stable on slow frames and high-refresh displays.
    const elapsed=Math.min(deltaMs/1000,.04), steps=Math.max(1,Math.ceil(elapsed/.008));
    const dt=elapsed/steps;
    for(let i=0;i<steps;i++) {
      soft.v+=(350*(soft.target-soft.x)-27*soft.v)*dt;
      soft.x+=soft.v*dt;
      const pull=Math.max(-15,Math.min(15,soft.v*.055));
      soft.lagV+=(280*(pull-soft.lag)-19*soft.lagV)*dt;
      soft.lag+=soft.lagV*dt;
    }
    if(Math.abs(soft.target-soft.x)<.025 && Math.abs(soft.v)<.08 && Math.abs(soft.lag)<.025 && Math.abs(soft.lagV)<.08) {
      soft.x=soft.target; soft.v=soft.lag=soft.lagV=0;
      gsap.ticker.remove(tickBall); ticking=false;
    }
    paintIndicator();
  }
  function moveIndicator(index, immediate=false) {
    index=Math.min(index,navButtons.length-1);
    if(index===navIndex && !immediate) return;
    const button=navButtons[index]; if(!button) return;
    const initial=navIndex<0; navIndex=index;
    soft.target=button.offsetLeft+button.offsetWidth/2;
    soft.y=button.offsetTop+button.offsetHeight/2; soft.r=button.offsetHeight/2;
    indicator.setAttribute('viewBox',`0 0 ${pageNav.clientWidth} ${pageNav.clientHeight}`);
    navButtons.forEach((b,i)=>{if(i===index)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
    if(initial || immediate || reducedMotion.matches) {
      gsap.ticker.remove(tickBall); ticking=false;
      soft.x=soft.target; soft.v=soft.lag=soft.lagV=0; paintIndicator();
    } else if(!ticking) { ticking=true; gsap.ticker.add(tickBall); }
  }
  new ResizeObserver(()=>moveIndicator(Math.max(0,navIndex),true)).observe(pageNav);
  reducedMotion.addEventListener('change',()=>moveIndicator(Math.max(0,navIndex),true));
  moveIndicator(0,true);


  // Only page boundaries are destinations; never leave a reading stop between pages.
  function measureStops() {
    const maxScroll=Math.max(0,document.documentElement.scrollHeight-window.innerHeight);
    stops=[...new Set([...document.querySelectorAll('[data-page]')].map(page=>
      Math.round(Math.min(page.getBoundingClientRect().top+window.scrollY,maxScroll))))];
    if(!scripted) current=nearestStop();
  }
  function nearestStop() {
    return stops.reduce((best, y, i) => Math.abs(y - window.scrollY) < Math.abs(stops[best] - window.scrollY) ? i : best, 0);
  }
  function cancelJump() {
    tween?.kill();
    tween = null;

    scripted = false;
    document.documentElement.classList.remove('scripted-scroll');
  }
  function goTo(index) {
    if (dialog.open) return;
    const fusionIndex=stops.length-1;
    const curtainTransition=index===fusionIndex || (index===fusionIndex-1 && window.scrollY>=stops[fusionIndex-1]-1);
    const velocity=curtainFlight.active?curtainFlight.pixelsPerSecond:0;
    cancelJump();
    current = Math.max(0, Math.min(stops.length - 1, index));
    scripted = true;
    const destinationCenter=stops[current]+window.innerHeight/2;
    const destinationPage=[...document.querySelectorAll('[data-page]')].findIndex(page=>{
      const top=page.getBoundingClientRect().top+window.scrollY;
      return destinationCenter>=top && destinationCenter<top+page.offsetHeight;
    });
    moveIndicator(Math.max(0,destinationPage));
    window.preparePaperSlide?.(destinationPage);
    if(document.querySelectorAll('[data-page]')[destinationPage]?.id==='probrope-formulations')window.formulationNavigation?.prepare(Math.sign(stops[current]-window.scrollY));
    // Every gesture goes directly to a complete page, including mid-flight reversals.
    document.documentElement.classList.add('scripted-scroll');
    if(curtainTransition && !reducedMotion.matches){
      animateCurtain(current===fusionIndex?1:0,velocity);
      return;
    }
    tween = gsap.to(window, {
      scrollTo: { y: stops[current], autoKill: false },
      duration: reducedMotion.matches ? 0 : 0.65,
      ease: 'power2.out',
      overwrite: true,
      onComplete: () => {cancelJump();activatePage();},
      onInterrupt: () => { scripted=false; document.documentElement.classList.remove('scripted-scroll'); }
    });
  }
  function activatePage() {
    const fusion=document.querySelector('.fusion-backdrop');
    const lift=Math.max(0,window.scrollY-(stops[stops.length-2] ?? Infinity));
    pageNav.style.translate=`0 ${-lift}px`;
    pageNav.inert=lift>=window.innerHeight-1;
    pageNav.setAttribute('aria-hidden',String(pageNav.inert));
    revealFusionContent(lift);
    const revealed=lift>1;
    fusion?.classList.toggle('is-revealed',revealed);
    if(fusion){fusion.inert=!revealed;fusion.setAttribute('aria-hidden',String(!revealed));}
    const pages = [...document.querySelectorAll('[data-page]')];
    const center = window.scrollY + window.innerHeight / 2;
    let index = pages.findIndex(page => {
      const top = page.getBoundingClientRect().top + window.scrollY;
      return center >= top && center < top + page.offsetHeight;
    });
    if (index < 0) index = 0;
    document.body.classList.toggle('on-slide', index > 0);
    if (!scripted) {moveIndicator(index);window.preparePaperSlide?.(index);}
    window.activatePaperSlide?.(index);
  }
  document.querySelectorAll('[data-goto]').forEach(button => button.addEventListener('click', (event) => {
    event.preventDefault();
    moveIndicator(Number(button.dataset.goto));
    const page = document.querySelectorAll('[data-page]')[Number(button.dataset.goto)];
    const y = Math.round(page.getBoundingClientRect().top + window.scrollY);
    goTo(stops.reduce((best,stop,i) => Math.abs(stop-y)<Math.abs(stops[best]-y)?i:best,0));
  }));
  const headingPages=[...document.querySelectorAll('[data-page]:not(.paper-page):not(.fusion-page)')];
  const formulaPage=document.querySelector('#probrope-formulations');
  formulaPage?.prepend(formulaPage.querySelector('.formulation-heading'));
  headingPages.forEach(page=>{const heading=page.querySelector(':scope > h1,:scope > h2,:scope > .formulation-heading');if(heading){heading.classList.add('page-heading');}});
  window.alignPaperHeadings=()=>headingPages.forEach(page=>{const heading=page.querySelector(':scope > .page-heading');if(heading)page.style.setProperty('--page-heading-height',heading.offsetHeight+'px');});
  const headingObserver=new ResizeObserver(()=>{window.alignPaperHeadings();document.dispatchEvent(new Event('paper-layout'));});
  headingPages.forEach(page=>{const heading=page.querySelector(':scope > .page-heading');if(heading)headingObserver.observe(heading);});
  window.alignPaperHeadings();
  const overview=document.querySelector('.paper-page');
  const overviewFit=document.createElement('div'); overviewFit.className='paper-fit';
  const overviewContent=document.createElement('div'); overviewContent.className='paper-content';
  while(overview.firstChild) overviewContent.append(overview.firstChild);
  overviewFit.append(overviewContent); overview.append(overviewFit);
  let overviewFitKey='';
  function fitOverview() {
    const css=getComputedStyle(overview);
    const available=Math.max(1,overview.clientHeight-parseFloat(css.paddingTop)-parseFloat(css.paddingBottom));
    const compact=matchMedia('(max-width:760px), (max-width:900px) and (orientation:portrait)').matches;
    const key=()=>[overview.clientWidth,available,overviewContent.offsetHeight,compact,document.fonts.status].join(':');
    if(key()===overviewFitKey)return;
    overviewContent.style.transformOrigin=compact?'top left':'top center';
    overviewContent.style.width='100%';
    let scale=Math.min(1,available/Math.max(1,overviewContent.offsetHeight));
    if(compact && scale<1){
      let low=scale,high=1;
      for(let i=0;i<10;i++){
        const candidate=(low+high)/2;
        overviewContent.style.width=`${100/candidate}%`;
        if(overviewContent.offsetHeight*candidate<=available)low=candidate;
        else high=candidate;
      }
      scale=low;
      overviewContent.style.width=`${100/scale}%`;
    }
    overviewContent.style.transform=`scale(${scale})`;
    overviewFit.style.height=`${overviewContent.offsetHeight*scale}px`;
    overviewFitKey=key();
    measureStops();
  }
  const overviewObserver=new ResizeObserver(fitOverview);
  overviewObserver.observe(overview); overviewObserver.observe(overviewContent);
  document.fonts.ready.then(fitOverview);
  fitOverview();
  measureStops();
  document.documentElement.classList.add('paged');
  const linkedPage=document.getElementById(location.hash.slice(1))?.closest('[data-page]');
  if(linkedPage){
    current=[...document.querySelectorAll('[data-page]')].indexOf(linkedPage);
    window.scrollTo(0,stops[current]);
    activatePage();
  }
  // One wheel/trackpad gesture means one page. Inertia never queues extra pages.
  // A direction reversal retargets immediately, even while the animation is running.
  function nativeCanScroll(target,direction) {
    const region=target.closest?.('[data-native-scroll]');
    if(!region) return false;
    return direction<0 ? region.scrollTop>1 : region.scrollTop+region.clientHeight<region.scrollHeight-1;
  }
  function stepPage(direction,useChapters=true) {
    const index=scripted?current:nearestStop();
    const page=document.querySelectorAll('[data-page]')[index];
    if(useChapters&&page?.id==='probrope-formulations'&&window.formulationNavigation?.step(direction))return 'chapters';
    goTo(index+direction);return 'page';
  }
  function activeScrollRegion(){return document.querySelectorAll('[data-page]')[scripted?current:nearestStop()]?.querySelector?.('[data-native-scroll]');}
  // Wheel events do not expose physical release. Keep a directional latch, but
  // recognize a short break in the event cadence or a restart after near-rest.
  // Mere acceleration during a continuous roll never releases the latch.
  const wheelGesture={time:-Infinity,direction:0,owner:null,region:null,intent:0,intentDirection:0,intentTime:-Infinity,cadence:16,magnitude:0,peak:0,quietSince:null,started:-Infinity,relaunch:null};
  const streamGap=250;
  window.addEventListener('wheel',event=>{
    if(dialog.open || event.ctrlKey || event.metaKey || !event.deltaY) return;
    if(Math.abs(event.deltaY)<=Math.abs(event.deltaX)) return;
    const now=Number.isFinite(event.timeStamp)&&event.timeStamp>0?event.timeStamp:performance.now(),direction=Math.sign(event.deltaY);
    const magnitude=Math.abs(event.deltaY)*(event.deltaMode===1?16:event.deltaMode===2?window.innerHeight:1);
    const gap=now-wheelGesture.time;
    const pauseGap=Math.max(64,Math.min(140,wheelGesture.cadence*2.5));
    const quietLimit=Math.max(.75,Math.min(3,wheelGesture.peak*.035));
    // A gap followed only by a smaller inertia sample is not a new swipe.
    const resumed=gap>=pauseGap && magnitude>=Math.max(4,wheelGesture.peak*.15,wheelGesture.magnitude*.98);
    const restarted=wheelGesture.quietSince!==null && now-wheelGesture.quietSince>=48 && magnitude>=Math.max(4,quietLimit*3);
    // Captured physical input: a new contact interrupts a still-large inertial
    // tail with a short cadence break and a sharp drop, then ramps up again.
    // Confirm the ramp; a gap or a decaying sample alone must never unlock.
    const contactBreak=Math.max(20,wheelGesture.cadence*2.2);
    if(direction===wheelGesture.direction && now-wheelGesture.started>120 &&
       gap>=contactBreak && gap<=streamGap &&
       (magnitude<=Math.max(3,wheelGesture.magnitude*.6) || gap>=pauseGap)){
      wheelGesture.relaunch={time:now,low:magnitude};
    }
    const candidate=wheelGesture.relaunch;
    const relaunched=!!candidate && now>candidate.time && now-candidate.time<=100 &&
      magnitude>=Math.max(4,candidate.low*1.65,candidate.low+3);
    if(candidate && (now-candidate.time>100 || direction!==wheelGesture.direction))wheelGesture.relaunch=null;
    const fresh=!wheelGesture.owner || gap>streamGap || direction!==wheelGesture.direction || resumed || restarted || relaunched;
    if(fresh){
      // Ignore subpixel jitter, but accept meaningful reversals on the first event.
      wheelGesture.intent=direction===wheelGesture.intentDirection && now-wheelGesture.intentTime<=streamGap ? wheelGesture.intent+magnitude : magnitude;
      wheelGesture.intentDirection=direction;wheelGesture.intentTime=now;
      if(wheelGesture.intent<2){event.preventDefault();return;}
    }
    if(fresh){
      const region=activeScrollRegion() || event.target.closest?.('[data-native-scroll]');
      const internal=region && (direction<0?region.scrollTop>1:region.scrollTop+region.clientHeight<region.scrollHeight-1);
      Object.assign(wheelGesture,{owner:internal?'gallery':'page',region,direction,intent:0,peak:magnitude,quietSince:null,started:now,relaunch:null});
      if(!internal)wheelGesture.owner=stepPage(direction);
    }
    else wheelGesture.intent=0;
    if(gap>0 && gap<=60)wheelGesture.cadence=wheelGesture.cadence*.75+gap*.25;
    wheelGesture.peak=Math.max(wheelGesture.peak,magnitude);
    if(magnitude<=Math.max(.75,Math.min(3,wheelGesture.peak*.035))){
      if(wheelGesture.quietSince===null)wheelGesture.quietSince=now;
    } else wheelGesture.quietSince=null;
    wheelGesture.magnitude=magnitude;wheelGesture.time=now;
    if(wheelGesture.owner==='gallery'){
      const region=wheelGesture.region;
      const canScroll=direction<0?region.scrollTop>1:region.scrollTop+region.clientHeight<region.scrollHeight-1;
      if(canScroll){
        if(event.target.closest?.('[data-native-scroll]')===region)return;
        region.scrollTop+=direction*magnitude;
      }
    }
    event.preventDefault();
  },{passive:false,capture:true});
  let touchGesture;
  window.addEventListener('touchstart',event=>{
    const nativeRegion=event.target.closest?.('[data-native-scroll]');
    const region=activeScrollRegion() || nativeRegion;
    const isFormulation=document.querySelectorAll('[data-page]')[scripted?current:nearestStop()]?.id==='probrope-formulations';
    if(isFormulation)window.formulationNavigation?.pause();
    touchGesture=event.touches.length===1 ? {
      x:event.touches[0].clientX,y:event.touches[0].clientY,
      lastX:event.touches[0].clientX,lastY:event.touches[0].clientY,axis:null,isFormulation,reverseDistance:0,direction:0,owner:null,region,native:region===nativeRegion,
      canUp:!!region && region.scrollTop>1,
      canDown:!!region && region.scrollTop+region.clientHeight<region.scrollHeight-1
    } : null;
  },{passive:true});
  window.addEventListener('touchmove',event=>{
    if(!touchGesture || event.touches.length!==1 || dialog.open) return;
    const t=event.touches[0], gesture=touchGesture;
    const dx=gesture.x-t.clientX,dy=gesture.y-t.clientY;
    if(!gesture.axis){
      if(Math.max(Math.abs(dx),Math.abs(dy))<8){
        if((gesture.isFormulation || !gesture.native) && event.cancelable)event.preventDefault();
        return;
      }
      gesture.axis=Math.abs(dx)>Math.abs(dy)?'x':'y';
    }
    if(gesture.axis==='x'){
      if(!gesture.isFormulation)return;
      if(event.cancelable)event.preventDefault();
      const movement=gesture.lastX-t.clientX;
      gesture.lastX=t.clientX;
      const direction=Math.sign(movement)||gesture.direction;
      if(!gesture.owner){
        gesture.owner='chapters';gesture.direction=direction;
        window.formulationNavigation?.step(direction);
      }else{
        gesture.reverseDistance=direction!==gesture.direction?gesture.reverseDistance+Math.abs(movement):0;
        if(gesture.reverseDistance>=8){
          gesture.direction=direction;gesture.reverseDistance=0;
          window.formulationNavigation?.step(direction);
        }
      }
      return;
    }
    const movement=gesture.lastY-t.clientY;
    gesture.lastY=t.clientY;
    const direction=Math.sign(movement)||gesture.direction;
    if(!gesture.owner) {
      gesture.owner=(direction<0?gesture.canUp:gesture.canDown)?'gallery':'page';
      gesture.direction=direction;
      if(gesture.owner==='page') gesture.owner=stepPage(direction,false);
    } else if(gesture.owner!=='gallery') {
      // Measure from the turning point, not the initial touch position. Slow
      // reversals count too, and crossing the initial point cannot stall input.
      gesture.reverseDistance=direction!==gesture.direction ? gesture.reverseDistance+Math.abs(movement) : 0;
      if(gesture.reverseDistance>=8){
        gesture.direction=direction;gesture.reverseDistance=0;
        const region=activeScrollRegion();
        const internal=region && (direction<0?region.scrollTop>1:region.scrollTop+region.clientHeight<region.scrollHeight-1);
        if(internal){gesture.owner='gallery';gesture.region=region;gesture.native=event.target.closest?.('[data-native-scroll]')===region;}
        else gesture.owner=stepPage(direction,false);
      }
    }
    if(gesture.owner==='gallery') {
      const region=gesture.region;
      const canScroll=direction<0?region.scrollTop>1:region.scrollTop+region.clientHeight<region.scrollHeight-1;
      if(canScroll){if(gesture.native)return;region.scrollTop+=movement;}
    }
    if(event.cancelable) event.preventDefault();
  },{passive:false});
  for(const type of ['touchend','touchcancel']) window.addEventListener(type,()=>{touchGesture=null;if(!scripted)settled();},{passive:true});
  function settled() {
    if(touchGesture)return;
    if(!scripted){
      if(touchViewport.matches){
        const y=stops[current];
        if(Number.isFinite(y) && Math.abs(window.scrollY-y)>1)window.scrollTo(0,y);
      }else current=nearestStop();
    }
    activatePage();
  }
  window.addEventListener('scroll', () => {
    if (!scrollFrame) scrollFrame=requestAnimationFrame(() => {
      scrollFrame=null;
      activatePage();
    });
    clearTimeout(settleTimer);
    settleTimer=setTimeout(settled,120);
  }, {passive:true});
  window.addEventListener('scrollend', settled, {passive:true});
  window.addEventListener('keydown', event => {
    if (dialog.open || event.ctrlKey || event.metaKey || event.altKey || event.target.closest('a,button,input,textarea,select,[contenteditable]')) return;
    const down = ['ArrowDown', 'PageDown', ' '].includes(event.key);
    const up = ['ArrowUp', 'PageUp'].includes(event.key);
    if (!(down || up || event.key === 'Home' || event.key === 'End')) return;
    const nativeRegion=event.target.closest('[data-native-scroll]');
    const direction=up || (event.key===' ' && event.shiftKey)?-1:1;
    if(nativeRegion && (event.key==='Home' || event.key==='End' || nativeCanScroll(event.target,direction))) return;
    event.preventDefault();
    if (event.repeat) return;
    if (event.key === 'Home') goTo(0);
    else if (event.key === 'End') goTo(stops.length - 1);
    else stepPage(up || (event.key === ' ' && event.shiftKey) ? -1 : 1);
  });
  document.addEventListener('paper-layout', () => {
    const destination=current;
    measureStops();
    current=destination;
    if(!scripted && !touchGesture){window.scrollTo(0,stops[current]);activatePage();}
  });
  let resizeTimer;
  let layoutWidth=window.innerWidth,layoutHeight=window.innerHeight;
  window.addEventListener('resize', () => {
    if(window.innerWidth===layoutWidth && (touchViewport.matches || window.innerHeight===layoutHeight))return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const destination=current;
      layoutWidth=window.innerWidth;layoutHeight=window.innerHeight;
      cancelJump();
      lockTouchViewport();
      measureStops();
      current=destination;
      window.scrollTo(0,stops[current]);
      activatePage();
    }, 150);
  });
  document.fonts.ready.then(() => {
    measureStops();
    window.scrollTo(0,stops[current]);
    activatePage();
    if (!reducedMotion.matches) gsap.from('[data-reveal]', {y:14, opacity:0, duration:0.75, stagger:0.07, ease:'power2.out', clearProps:'transform,opacity'});
  });
})();
