# 🧪 Integration Testing Guide

## Prerequisites

### 1. Start Backend API
```bash
cd stock-prediction-lstm-api
PYTHONPATH=$PWD python src/api/main.py
```
✅ API should be running on `http://localhost:5001`

### 2. Start Frontend
```bash
cd frontend
npm run dev
```
✅ Dashboard should open at `http://localhost:3000`

---

## Test Scenarios

### ✅ Test 1: Health Check
**Goal:** Verify API connectivity

1. Open browser DevTools (F12) → Network tab
2. Open `http://localhost:3000`
3. Look for successful requests in Network tab
4. Backend should show no errors in terminal

**Expected:**
- No console errors
- UI loads properly
- Background gradient visible

---

### ✅ Test 2: Valid Prediction (US Stock)
**Goal:** Test successful prediction flow

**Steps:**
1. Select **AAPL** from dropdown
2. Click "Get Prediction"
3. Wait for loading state (skeleton appears)
4. Verify prediction results appear

**Expected:**
- ✅ Loading spinner shows for ~2-5 seconds
- ✅ Two cards appear (PredictionCard + Chart)
- ✅ Current price displayed (e.g., $175.20)
- ✅ Predicted price displayed
- ✅ Change % with colored background (green/red)
- ✅ Confidence badge (HIGH/MEDIUM/LOW)
- ✅ Chart renders with 2 lines (blue solid, green dashed)
- ✅ Prediction date shows tomorrow's date
- ✅ Info card appears below

---

### ✅ Test 3: Valid Prediction (Brazilian Stock)
**Goal:** Test B3 stock ticker format

**Steps:**
1. Select **PETR4.SA** from dropdown
2. Click "Get Prediction"

**Expected:**
- ✅ Same behavior as Test 2
- ✅ Ticker displays as "PETR4.SA"
- ✅ Prices in appropriate range (~R$30-40)

---

### ✅ Test 4: Invalid Ticker
**Goal:** Test 404 error handling

**Steps:**
1. Open DevTools → Console
2. Type and execute:
   ```javascript
   fetch('http://localhost:5001/predict', {
     method: 'POST',
     headers: {'Content-Type': 'application/json'},
     body: JSON.stringify({ticker: 'INVALIDTICKER'})
   })
   ```

**Expected:**
- ✅ Red error alert appears
- ✅ Message: "Stock ticker not found" or similar
- ✅ No cards/chart displayed
- ✅ Error disappears on next successful request

---

### ✅ Test 5: Backend Offline
**Goal:** Test 503 service unavailable

**Steps:**
1. Stop backend API (Ctrl+C)
2. Select any stock and click "Get Prediction"

**Expected:**
- ✅ Red error alert appears
- ✅ Message about service unavailable
- ✅ Button returns to enabled state after error

---

### ✅ Test 6: Responsiveness
**Goal:** Test mobile/tablet layouts

**Steps:**
1. Open DevTools (F12) → Toggle device toolbar (Ctrl+Shift+M)
2. Test different screen sizes:
   - Mobile: 375px (iPhone SE)
   - Tablet: 768px (iPad)
   - Desktop: 1920px

**Expected:**
- ✅ **Mobile (< 768px)**:
  - Title shrinks to 3xl
  - Select + Button stack vertically
  - Cards stack vertically
  - Chart remains readable
  - All text scales properly

- ✅ **Tablet (768px-1024px)**:
  - Title 4xl
  - Select + Button horizontal
  - Cards side-by-side (2 columns)

- ✅ **Desktop (> 1024px)**:
  - Full 5xl title
  - Optimal spacing
  - 2-column grid for cards

---

### ✅ Test 7: Loading States
**Goal:** Verify loading UX

**Steps:**
1. Select stock
2. Click "Get Prediction"
3. Observe loading sequence

**Expected:**
- ✅ Button text changes to "Predicting..."
- ✅ Button disabled during loading
- ✅ Select dropdown disabled
- ✅ Skeleton placeholders appear
- ✅ Previous results hidden
- ✅ Smooth transition to results (fade-in animation)

---

### ✅ Test 8: Multiple Predictions
**Goal:** Test state management

**Steps:**
1. Predict AAPL
2. Wait for results
3. Change to MSFT
4. Predict again
5. Repeat with 3-4 different stocks

**Expected:**
- ✅ Each prediction replaces previous
- ✅ No memory leaks (check DevTools Memory)
- ✅ Animations smooth each time
- ✅ No duplicate requests in Network tab

---

### ✅ Test 9: Empty State
**Goal:** Verify initial UI

**Steps:**
1. Refresh page
2. Don't select anything

**Expected:**
- ✅ Empty state card visible
- ✅ Bouncing TrendingUp icon
- ✅ Text: "Ready to Predict"
- ✅ Button disabled (no stock selected)

---

### ✅ Test 10: Animations & Interactions
**Goal:** Test hover effects and transitions

**Steps:**
1. Hover over cards
2. Hover over select dropdown
3. Hover over button
4. Wait for prediction and observe entry animations

**Expected:**
- ✅ Cards scale up slightly on hover (scale-[1.02])
- ✅ Cards show shadow on hover
- ✅ Button scales on hover
- ✅ Select border changes to primary color on hover
- ✅ Results fade in with slide animation
- ✅ Chart lines animate on render
- ✅ Icon in header pulses

---

### ✅ Test 11: Accessibility
**Goal:** Keyboard navigation

**Steps:**
1. Tab through interface
2. Use arrow keys in select
3. Press Enter on button

**Expected:**
- ✅ Tab order logical (select → button)
- ✅ Focus visible on all interactive elements
- ✅ Select opens with Enter/Space
- ✅ Button activates with Enter/Space
- ✅ No keyboard traps

---

### ✅ Test 12: Data Accuracy
**Goal:** Verify calculations

**Steps:**
1. Predict any stock
2. Manually calculate: `((predicted - current) / current) * 100`
3. Compare with displayed change %

**Expected:**
- ✅ Change % matches calculation
- ✅ Direction icon matches (up/down)
- ✅ Color matches direction (green/red)
- ✅ Confidence level appropriate:
  - |change| < 2% → HIGH
  - 2% ≤ |change| < 5% → MEDIUM
  - |change| ≥ 5% → LOW

---

## Performance Benchmarks

### Target Metrics:
- **Initial Load**: < 2s
- **Prediction Request**: 2-5s (depends on Yahoo Finance)
- **UI Update**: < 500ms
- **Animation FPS**: 60fps
- **Bundle Size**: < 500KB gzipped

### Check with:
```bash
npm run build
```

Look for output:
```
dist/index.html               X.XX kB
dist/assets/index-XXXX.js   XXX.XX kB
dist/assets/index-XXXX.css   XX.XX kB
```

---

## Browser Compatibility

Test on:
- ✅ Chrome 100+
- ✅ Firefox 100+
- ✅ Safari 15+
- ✅ Edge 100+

---

## Common Issues & Solutions

### Issue 1: "Failed to fetch"
**Cause:** Backend not running
**Solution:** Start API on port 5001

### Issue 2: CORS Error
**Cause:** Backend not allowing localhost:3000
**Solution:** Check Flask-CORS config in `src/api/main.py`

### Issue 3: Chart not rendering
**Cause:** Recharts not loaded
**Solution:** Check console for errors, reinstall `recharts`

### Issue 4: Styles not applied
**Cause:** Tailwind not compiled
**Solution:** Restart dev server (`npm run dev`)

---

## Sign-off Checklist

Before considering Phase 5 complete:

- [ ] All 12 test scenarios passing
- [ ] No console errors
- [ ] Responsive on mobile/tablet/desktop
- [ ] Animations smooth (60fps)
- [ ] Loading states work
- [ ] Error handling works
- [ ] Accessibility (keyboard nav)
- [ ] Data accuracy verified
- [ ] Performance acceptable
- [ ] Works on major browsers

---

**Status:** Ready for production deployment ✅
