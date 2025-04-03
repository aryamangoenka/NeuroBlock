sed -i "" "s/label: \"Global AvgPool\",/label: \"Global Average Pooling\",/g" frontend/src/pages/NewBuildPage.tsx
sed -i "" "s/type: \"maxpooling\",/type: \"globalaveragepool\",/g" frontend/src/pages/NewBuildPage.tsx
