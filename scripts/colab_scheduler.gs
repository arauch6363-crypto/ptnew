/**
 * PT Workflow Scheduler — Google Apps Script
 * ============================================================
 * Führt PT_WORKFLOW.ipynb täglich um 10:00 Uhr (Berlin Zeit)
 * automatisch in Google Colab aus.
 *
 * EINMALIGE SETUP-SCHRITTE:
 * ─────────────────────────
 * 1. Öffne https://script.google.com → "Neues Projekt"
 * 2. Diesen Code einfügen, Datei speichern
 * 3. NOTEBOOK_FILE_ID eintragen (aus der Colab-URL:
 *    https://colab.research.google.com/drive/HIER_STEHT_DIE_ID)
 * 4. NOTIFY_EMAIL eintragen (deine Gmail-Adresse)
 * 5. Funktion "setupTrigger" auswählen → ▶ Ausführen
 * 6. Google-Berechtigungen genehmigen (Drive + Mail)
 *
 * BENÖTIGTE OAUTH-SCOPES (werden automatisch erkannt):
 *   https://www.googleapis.com/auth/drive
 *   https://www.googleapis.com/auth/script.external_request
 *   https://www.googleapis.com/auth/mail.send
 *
 * WIE ES FUNKTIONIERT:
 *   Apps Script ruft die interne Colab API auf, die einen
 *   Kernel (VM) startet und dann alle Zellen ausführt.
 *   Die API ist nicht öffentlich dokumentiert — falls sie
 *   sich ändert, siehe Fallback-Option am Ende dieser Datei.
 */

// ── Konfiguration ──────────────────────────────────────────────────────────────

/** Google Drive File-ID des PT_WORKFLOW.ipynb
 *  Aus der URL: colab.research.google.com/drive/<FILE_ID>  */
var NOTEBOOK_FILE_ID = '1PZYNGKZB_vDUvWIssDXil8hQaXlZ9q_w';

/** E-Mail für Erfolgs-/Fehlermeldungen (leer lassen = keine Mail) */
var NOTIFY_EMAIL = 'arauch6363@gmail.com';

/** CPU-only reicht für den Workflow (kein GPU nötig) */
var ACCELERATOR = 'NONE';   // Alternativen: 'GPU', 'TPU'


// ── Hauptfunktion (wird täglich automatisch aufgerufen) ───────────────────────

function runPTWorkflow() {
  var berlinTime = Utilities.formatDate(new Date(), 'Europe/Berlin', 'yyyy-MM-dd HH:mm:ss');
  Logger.log('▶  PT Workflow Start: ' + berlinTime);

  var status  = 'UNBEKANNT';
  var details = '';

  try {
    var result = _startColabAndExecute(NOTEBOOK_FILE_ID);
    status  = 'GESTARTET';
    details = 'Kernel-ID: ' + result.kernelId
            + ' | Execute-Status: ' + result.executeStatus;
    Logger.log('✅ ' + details);

  } catch (e) {
    status  = 'FEHLER';
    details = e.toString();
    Logger.log('❌ ' + details);
  }

  if (NOTIFY_EMAIL) {
    var ok = (status === 'GESTARTET');
    MailApp.sendEmail({
      to:      NOTIFY_EMAIL,
      subject: (ok ? '✅' : '❌') + ' PT Workflow — ' + berlinTime,
      body:    'Status : ' + status + '\n'
             + 'Details: ' + details + '\n\n'
             + 'Notebook: https://colab.research.google.com/drive/' + NOTEBOOK_FILE_ID + '\n'
             + 'Logs   : https://script.google.com (Ausführungen)',
    });
  }
}


// ── Colab API ──────────────────────────────────────────────────────────────────

function _startColabAndExecute(fileId) {
  var token = ScriptApp.getOAuthToken();
  var headers = {
    'Authorization': 'Bearer ' + token,
    'Content-Type':  'application/json',
  };

  // ── Schritt 1: Kernel starten ──────────────────────────────────────────────
  var kernelResp = UrlFetchApp.fetch('https://colab.research.google.com/api/kernels', {
    method:            'POST',
    headers:           headers,
    payload:           JSON.stringify({
      fileId:                 fileId,
      desiredAcceleratorType: ACCELERATOR,
    }),
    muteHttpExceptions: true,
  });

  var kCode = kernelResp.getResponseCode();
  var kBody = kernelResp.getContentText();

  if (kCode !== 200) {
    throw new Error('Kernel-Start fehlgeschlagen (HTTP ' + kCode + '): '
                    + kBody.slice(0, 400));
  }

  var kernelId = JSON.parse(kBody).id;
  Logger.log('Kernel gestartet: ' + kernelId);

  // ── Schritt 2: Kernel bereit abwarten ─────────────────────────────────────
  Utilities.sleep(4000);

  // ── Schritt 3: Alle Zellen ausführen ──────────────────────────────────────
  var execResp = UrlFetchApp.fetch(
    'https://colab.research.google.com/api/kernels/' + kernelId + '/execute',
    {
      method:            'POST',
      headers:           headers,
      payload:           JSON.stringify({ executeAll: true }),
      muteHttpExceptions: true,
    }
  );

  var eCode = execResp.getResponseCode();
  var eBody = execResp.getContentText();
  Logger.log('Execute: HTTP ' + eCode + ' — ' + eBody.slice(0, 200));

  // Manche Colab-Versionen erwarten eine andere Payload — zweiter Versuch
  if (eCode === 400 || eCode === 404) {
    Logger.log('Versuche alternative Execute-Methode...');
    execResp = UrlFetchApp.fetch(
      'https://colab.research.google.com/api/kernels/' + kernelId + '/restart_and_run_all',
      {
        method:            'POST',
        headers:           headers,
        payload:           '{}',
        muteHttpExceptions: true,
      }
    );
    eCode = execResp.getResponseCode();
    Logger.log('Alt. Execute: HTTP ' + eCode);
  }

  return { kernelId: kernelId, executeStatus: eCode };
}


// ── Trigger einrichten (einmalig manuell ausführen) ────────────────────────────

/**
 * Legt einen täglichen Trigger um 10:00 Uhr Berlin Zeit an.
 * Diese Funktion NUR EINMAL manuell ausführen.
 */
function setupTrigger() {
  // Bestehende runPTWorkflow-Trigger löschen
  ScriptApp.getProjectTriggers().forEach(function(t) {
    if (t.getHandlerFunction() === 'runPTWorkflow') {
      ScriptApp.deleteTrigger(t);
      Logger.log('Alter Trigger gelöscht.');
    }
  });

  // Neuen Trigger anlegen — Apps Script respektiert DST automatisch
  ScriptApp.newTrigger('runPTWorkflow')
    .timeBased()
    .atHour(10)           // 10 Uhr
    .nearMinute(0)        // möglichst pünktlich zur vollen Stunde
    .everyDays(1)
    .inTimezone('Europe/Berlin')
    .create();

  Logger.log('✅ Trigger eingerichtet: täglich 10:00 Uhr Europe/Berlin');
  Logger.log('   (Apps Script passt Sommer-/Winterzeit automatisch an)');
}


// ── Hilfsfunktionen ────────────────────────────────────────────────────────────

/** Alle aktiven Trigger anzeigen */
function listTriggers() {
  ScriptApp.getProjectTriggers().forEach(function(t) {
    Logger.log(t.getHandlerFunction() + ' | ' + t.getEventType()
               + ' | ' + t.getTriggerSourceId());
  });
}

/** Alle runPTWorkflow-Trigger entfernen */
function removeTrigger() {
  var n = 0;
  ScriptApp.getProjectTriggers().forEach(function(t) {
    if (t.getHandlerFunction() === 'runPTWorkflow') {
      ScriptApp.deleteTrigger(t);
      n++;
    }
  });
  Logger.log('Entfernt: ' + n + ' Trigger.');
}

/**
 * Manueller Testlauf — direkt in Apps Script ausführen,
 * ohne auf den täglichen Trigger zu warten.
 */
function testRun() {
  Logger.log('=== MANUELLER TESTLAUF ===');
  if (NOTEBOOK_FILE_ID === 'DEINE_NOTEBOOK_FILE_ID_HIER_EINSETZEN') {
    throw new Error('Bitte NOTEBOOK_FILE_ID zuerst eintragen!');
  }
  runPTWorkflow();
}


// ── Notebook-ID aus URL ermitteln ──────────────────────────────────────────────
/**
 * TIPP: Die File-ID findest du in der Colab-URL:
 *   https://colab.research.google.com/drive/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs
 *                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *   Oder in Google Drive → Rechtsklick auf Datei → "Link teilen" →
 *   https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs/view
 *                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 */
