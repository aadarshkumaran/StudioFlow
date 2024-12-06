// background.js
chrome.runtime.onInstalled.addListener(() => {
  console.log('YouTube Studio Enhancer extension installed');
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Received message in background script:', request);

  if (request.type === 'FETCH_CAPTIONS') {
    console.log('Processing captions fetch request:', request.videoId);
    fetchYoutubeCaptions(request.videoId)
      .then(response => {
        console.log('Captions fetch response:', response);
        sendResponse({ success: true, data: response });
      })
      .catch(error => {
        console.error('Captions fetch error:', error);
        sendResponse({ success: false, error: error.message });
      });
    return true; // Will respond asynchronously
  }

  if (request.type === 'GENERATE_CONTENT') {
    console.log('Processing content generation request:', request.data);
    handleContentGeneration(request.data)
      .then(response => {
        console.log('Content generation response:', response);
        sendResponse({ success: true, data: response });
      })
      .catch(error => {
        console.error('Content generation error:', error);
        sendResponse({ success: false, error: error.message });
      });
    return true; // Will respond asynchronously
  }

  if (request.type === 'ENHANCE_CONTENT') {
    console.log('Processing content enhancement request:', request.data);
    handleContentEnhancement(request.data)
      .then(response => {
        console.log('Content enhancement response:', response);
        sendResponse({ success: true, data: response });
      })
      .catch(error => {
        console.error('Content enhancement error:', error);
        sendResponse({ success: false, error: error.message });
      });
    return true; // Will respond asynchronously
  }
});

async function fetchYoutubeCaptions(videoId) {
  try {
    console.log('Fetching captions for video ID:', videoId);
    
    const videoPageResponse = await fetch(`https://www.youtube.com/watch?v=${videoId}`);
    const pageText = await videoPageResponse.text();
    
    const playerResponseMatch = pageText.match(/var ytInitialPlayerResponse\s*=\s*({.*?});/);
    if (!playerResponseMatch) {
      throw new Error('Could not find player response');
    }

    const playerResponse = JSON.parse(playerResponseMatch[1]);
    const captionTracks = playerResponse.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];
    
    const autoCaptions = captionTracks
      .filter(track => track.kind === 'asr')
      .map(track => ({
        languageCode: track.languageCode,
        baseUrl: track.baseUrl
      }));

    if (autoCaptions.length === 0) {
      throw new Error('No auto-generated captions found');
    }

    const captionsResponse = await fetch(autoCaptions[0].baseUrl);
    const captionsXml = await captionsResponse.text();

    // Parse XML manually without using DOMParser
    const texts = [];
    const xmlRegex = /<text start="([\d.]+)" dur="([\d.]+)">(.*?)<\/text>/g;
    let match;

    while ((match = xmlRegex.exec(captionsXml)) !== null) {
      texts.push({
        start: parseFloat(match[1]),
        duration: parseFloat(match[2]),
        text: match[3].replace(/&amp;/g, '&')
                     .replace(/&lt;/g, '<')
                     .replace(/&gt;/g, '>')
                     .replace(/&quot;/g, '"')
                     .replace(/&#39;/g, "'")
                     .trim()
      });
    }

    return texts;
  } catch (error) {
    console.error('Error fetching YouTube captions:', error);
    throw error;
  }
}

async function handleContentGeneration({ prompt, choice, cc }) {
  console.log('Handling content generation:', { 
    prompt, 
    choice,
    ccLength: cc ? cc.length : 0,
    ccPreview: cc ? cc.substring(0, 100) + '...' : 'No captions'
  });

  try {
    const response = await fetch('http://44.202.239.160:8080/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: prompt, choice, cc })
    });

    console.log('Fetch response status:', response.status);

    if (response.ok) {
      const data = await response.json();
      console.log('Processed content data:', data);
      return data.result;
    } else {
      const errorText = await response.text();
      console.error('Error generating content:', errorText);
      throw new Error(`Error generating content: ${errorText}`);
    }
  } catch (error) {
    console.error('Catch block - Error generating content:', error);
    throw error;
  }
}

async function handleContentEnhancement({ text, contentType, userPrompt }) {
  console.log('Handling content enhancement:', { text, contentType, userPrompt });

  try {
    const response = await fetch('http://44.202.239.160:8080/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, contentType, user_prompt: userPrompt })
    });

    console.log('Fetch response:', response);

    if (response.ok) {
      const data = await response.json();
      console.log('Enhanced content data:', data);
      return data.result;
    } else {
      const errorText = await response.text();
      console.error('Error enhancing content:', errorText);
      throw new Error(`Error enhancing content: ${errorText}`);
    }
  } catch (error) {
    console.error('Catch block - Error enhancing content:', error);
    throw error;
  }
}

console.log('YouTube Studio Enhancer background script loaded.');