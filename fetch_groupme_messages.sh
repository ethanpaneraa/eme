#!/usr/bin/env bash
#
# fetch_groupme_messages.sh
# Pulls *all* messages from a GroupMe group and writes one JSON object per line.
#
# Usage: ./fetch_groupme_messages.sh GROUP_ID ACCESS_TOKEN OUTPUT_FILE

GROUP_ID="$1"
TOKEN="$2"
OUTFILE="$3"

# make sure we start fresh
> "$OUTFILE"

# initial call: no before_id
before_id=""

while :; do
  # build args
  args=( -G "https://api.groupme.com/v3/groups/${GROUP_ID}/messages"
         --data-urlencode "token=${TOKEN}"
         --data-urlencode "limit=100" )
  [[ -n "$before_id" ]] && args+=( --data-urlencode "before_id=${before_id}" )

  # fetch a batch
  resp=$(curl -s "${args[@]}")
  msgs=$(jq -c '.response.messages[]' <<<"$resp")
  count=$(jq '.response.messages | length' <<<"$resp")

  # break if no more
  if (( count == 0 )); then
    echo "Done—no more messages."
    break
  fi

  # append each message as a line
  echo "$msgs" >> "$OUTFILE"

  # get the oldest ID for next loop
  before_id=$(jq -r '.response.messages[-1].id' <<<"$resp")
  echo "Fetched $count messages; next before_id=$before_id"
done

echo "All messages saved to $OUTFILE"
