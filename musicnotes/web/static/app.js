// Upload, follow the job, show the result.
(function () {
  const $ = (id) => document.getElementById(id);
  const form = $('setup'), fileInput = $('file'), drop = $('drop');
  const NAMES = { guitar: 'Guitar', piano: 'Piano', drums: 'Drums', melody: 'Melody instrument' };
  let file = null, pollTimer = 0, previewUrl = null;

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  function setFile(f) {
    if (!f) return;
    if (!f.type.startsWith('video/') && !/\.(mp4|mov|m4v|webm|mkv|avi|3gp)$/i.test(f.name)) {
      showError('That is not a video file. Choose an MP4, MOV or WebM.');
      return;
    }
    file = f;
    showError('');
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    const v = $('preview');
    v.src = previewUrl;
    v.hidden = false;
    $('dropEmpty').hidden = true;
    v.onloadedmetadata = () => { $('filename').textContent = `${f.name} · ${fmt(v.duration)} · ${(f.size / 1048576).toFixed(1)} MB`; };
    $('filename').hidden = false;
    $('start').disabled = false;
  }

  function showError(msg) { $('formError').hidden = !msg; $('formError').textContent = msg; }

  fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); setFile(e.dataTransfer.files[0]); });

  function stage(name, progress, message) {
    const order = ['upload', 'listen', 'look', 'render', 'done'];
    const idx = order.indexOf(name);
    document.querySelectorAll('#stages li').forEach((li) => {
      const i = order.indexOf(li.dataset.stage);
      li.classList.toggle('done', i < idx);
      li.classList.toggle('active', i === idx);
    });
    $('bar').style.width = `${Math.round(progress * 100)}%`;
    $('status').textContent = message;
  }

  function showPanel(which) {
    $('resultEmpty').hidden = which !== 'empty';
    $('progress').hidden = which !== 'progress';
    $('done').hidden = which !== 'done';
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!file) return;
    const data = new FormData();
    data.append('file', file);
    for (const el of form.elements) {
      if (!el.name || el.type === 'file') continue;
      if (el.type === 'checkbox') data.append(el.name, el.checked ? 'true' : 'false');
      else if (el.type === 'radio') { if (el.checked) data.append(el.name, el.value); }
      else data.append(el.name, el.value);
    }
    $('start').disabled = true;
    showError('');
    showPanel('progress');
    stage('upload', 0, 'Uploading…');
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) stage('upload', 0.03 * ev.loaded / ev.total, `Uploading… ${Math.round((100 * ev.loaded) / ev.total)}%`); };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
      if (xhr.status >= 400) {
        showPanel('empty');
        showError(body.detail || 'The upload failed. Try again.');
        $('start').disabled = false;
        return;
      }
      poll(body.id);
    };
    xhr.onerror = () => { showPanel('empty'); showError('The upload failed. Check your connection and try again.'); $('start').disabled = false; };
    xhr.send(data);
  });

  function poll(id) {
    clearTimeout(pollTimer);
    fetch(`/api/jobs/${id}`).then((r) => r.json()).then((job) => {
      if (job.status === 'failed') {
        showPanel('empty');
        showError(job.error || 'Processing failed.');
        $('start').disabled = false;
        return;
      }
      if (job.status === 'done') return finish(job);
      stage(job.stage === 'queued' ? 'listen' : job.stage, job.progress, `${job.message}… ${Math.round(job.progress * 100)}%`);
      pollTimer = setTimeout(() => poll(id), 1000);
    }).catch(() => { pollTimer = setTimeout(() => poll(id), 2000); });
  }

  function finish(job) {
    stage('done', 1, 'Ready');
    showPanel('done');
    const v = $('output');
    v.src = `/api/jobs/${job.id}/video`;
    $('download').href = `/api/jobs/${job.id}/video?download=true`;
    const s = job.summary;
    const facts = $('facts');
    facts.textContent = '';
    const add = (k, val) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = val;
      facts.append(dt, dd);
    };
    add('Instrument', `${NAMES[s.instrument]} (${s.why})`);
    const geo = s.geometry;
    if (s.instrument === 'guitar') add('Fretboard', geo.neck ? (geo.frets_fitted > 0.3 ? 'Neck and frets mapped from the video' : 'Neck found; fret positions estimated from typical proportions') : 'Neck not found; effects follow the guitar');
    if (s.instrument === 'piano') add('Keys', geo.keyboard ? 'Keyboard mapped from the black-key pattern' : 'Keys not mapped; showing a keyboard strip instead');
    if (s.instrument === 'drums') add('Drum kit', geo.drum_pieces.length ? `Placed: ${geo.drum_pieces.join(', ')}` : 'Kit not found; hits shown in the frame');
    if (s.instrument === 'drums') add('Hits', `${s.drums.kick} kicks · ${s.drums.snare} snares · ${s.drums.hat} hi-hats`);
    else add('Notes', `${s.notes} notes${s.chords.length ? ` · ${s.chords.length} chords` : ''}`);
    add('Tempo', `${s.tempo} BPM`);
    add('Took', `${s.seconds} s for a ${fmt(s.duration)} clip`);
    const chips = $('chips');
    chips.textContent = '';
    for (const c of s.chords) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = c.name;
      const t = document.createElement('small');
      t.textContent = fmt(c.t0);
      b.append(t);
      b.addEventListener('click', () => { v.currentTime = c.t0; v.play().catch(() => {}); });
      chips.append(b);
    }
    $('start').disabled = false;
  }

  $('again').addEventListener('click', () => { showPanel('empty'); window.scrollTo({ top: 0, behavior: 'smooth' }); });
})();
